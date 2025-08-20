from typing import Callable
from pathlib import Path
import csv
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

# from diffusion_unet import UNet
from diffusers import UNet2DModel

from utils.schedulers import Scheduler


class Diffusion:

    def __init__(self, scheduler: Scheduler, noise_like: Callable):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scheduler = scheduler
        self.noise_like = noise_like

        self.coeffs = self._create_coeffs_from_betas(scheduler.betas)
        self.T = scheduler.T

        self.model = UNet2DModel().to(self.device)


    def _create_coeffs_from_betas(self, betas: torch.Tensor) -> dict[str, torch.Tensor]:
        coeffs = dict()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # for forward process q(x_t | x_t-1)
        coeffs["sqrt_one_minus_b"] = torch.sqrt(1 - betas)
        coeffs["sqrt_b"] = torch.sqrt(betas) # also for reverse process (substituted as sigmas)
        
        # for forward process q(x_t | x_0)
        coeffs["sqrt_a_bar"] = torch.sqrt(alphas_bar)
        coeffs["sqrt_one_minus_a_bar"] = torch.sqrt(1 - alphas_bar)

        # for reverse process p_theta(x_t-1 | x_t)
        coeffs["image_coeff"] = torch.sqrt(1 / alphas)
        coeffs["noise_coeff"] = ((1 - alphas) / torch.sqrt(1 - alphas_bar))

        return coeffs
    

    def _get_linear_betas(self, T: int, start: float = 0.0001, end: float = 0.02) -> torch.Tensor:
        # DDPM paper defaults: T=1000, start=0.0001, end=0.02
        betas = torch.linspace(start, end, T).to(self.device)
        return betas


    def get_forward_process_list(self, x_0: torch.Tensor) -> list[torch.Tensor]:
        """
        Samples a new image from q step by step
        Prepares list for visualization of forward process
        Returns the list with images from complete forward diffusion process
        x_0: original image, shape: (batch_size, C, H, W)
        """

        x_t = x_0
        images = [x_t]
        c = self.coeffs

        for t in range(self.T):
            noise = self.noise_like(x_t).to(self.device)
            x_t = c['sqrt_one_minus_b'][t] * x_t + c['sqrt_b'][t] * noise   # sample from q(x_t|x_t-1)
            images.append(x_t)
        
        return images


    def sample_forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a new image from q in one step at timestep t
        Used for training the model
        Returns noisy image and the applied noise at timestep t

        x_0: original image, shape: (batch_size, C, H, W)
        t: timestep, shape: (batch_size, )
        """
        assert 0 <= t.min() and t.max() < self.T
        assert x_0.shape[0] == t.shape[0]
        
        t = t.view(t.shape[0], 1, 1, 1)

        c = self.coeffs
        noise = self.noise_like(x_0).to(self.device)

        x_t = c["sqrt_a_bar"][t] * x_0 + c["sqrt_one_minus_a_bar"][t] * noise   # sample from q(x_t | x_0)

        return x_t, noise


    @torch.no_grad()
    def sample_reverse_process(self, x_t: torch.Tensor, t: int, e_t: torch.Tensor) -> torch.Tensor:
        """
        Samples a new image from p_theta (one iteration of Algorithm 2 in DDPM paper)
        Used for inference
        Returns slightly denoised image - x_(t-1) 

        x_t: noisy image, shape: (batch_size, C, H, W)
        t: timestep
        e_t: predicted noise, shape: (batch_size, C, H, W)
        """
        assert 0 <= t < self.T
        assert x_t.shape == e_t.shape

        c = self.coeffs
        
        if t > 0:
            sigma_t = c['sqrt_b'][t]   # setting sigma_t**2 to B_t works, variances may be learned by nn.
        else:
            sigma_t = 0
        
        u_t = c['image_coeff'][t] * (x_t - c['noise_coeff'][t] * e_t)
        new_x = u_t + sigma_t * self.noise_like(x_t).to(self.device)   # sample from p_theta(x_t-1 | x_t)

        return new_x


    def get_loss(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform forward process, predict noise, get MSE
        """

        x_noisy, noise = self.sample_forward_process(x_0, t)
        noise_pred = self.model(x_noisy, t).sample  # .sample, because we use diffusers unet
        loss = F.mse_loss(noise, noise_pred)

        return loss


    def train(self, dataloader: DataLoader, lr: float, epochs: int, patience: int = 10, lr_patience: int = 5, factor: float = 0.5, 
              experiment_path: Path = None, checkpoint_path: Path = None) -> None:
        """
        Train UNet model.

        Args:
            dataloader: PyTorch DataLoader providing training data.
            lr: Learning rate.
            epochs: Maximum number of epochs.
            patience: Number of epochs to wait for loss improvement before stopping.
            lr_patience: Number of epochs without improvement before reducing LR.
            factor: Factor to multiply LR when loss plateaus (e.g., 0.5 reduces LR by half).
            model_path: Name of checkpoint to which the modell will be saved
        """

        def log_epoch(experiment_path: Path, log_row: dict):
            timestamp = time.strftime("%d-%m-%y-%H-%M-%S", time.localtime())
            path = Path(experiment_path, f"training_logs-{timestamp}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", newline="") as f:
                writer = csv.writer(f)
                if epoch == 0:  writer.writerow(log_row.keys())
                writer.writerow(log_row.values())

        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=lr_patience)

        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for (x, _) in dataloader:
                x = x.to(self.device)
                optimizer.zero_grad()
                t = torch.randint(low=0, high=self.T, size=(x.shape[0],)).to(self.device)
                loss = self.get_loss(x, t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch} | Avg loss: {avg_loss} | LR: {current_lr} ")

            # Adjust learning rate
            lr_scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print("New min loss. Model saved.")
            else:
                no_improve_count += 1
            
            #Log epoch info to file
            log_row = {"epoch": epoch,
                        "avg_loss": avg_loss,
                        "best_loss": best_loss,
                        "current_lr": current_lr,
                        "no_improve_count": no_improve_count}
            log_epoch(experiment_path=experiment_path, log_row=log_row)
            
            # Early stopping check
            if no_improve_count >= patience:
                print(f"Early stopping after {epoch+1}. No improvement for {patience} epochs.")
                break


    
    @torch.no_grad()
    def get_backward_process_list(self, T: int, batch_size=1, image_shape: tuple = (3, 32, 32), checkpoint_path=None) -> list[torch.Tensor]:
        """
        Returns list of denoising process. Each element is a batch of images at timestep t
        Generate image from noise
        (Algorithm 2 in DDPM paper)
        """

        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        tensor = torch.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        x_t = self.noise_like(tensor).to(self.device)
        images = [x_t]

        for t in range(0, T)[::-1]:
            t = torch.full((1,), t).to(self.device)
            e_t = self.model(x_t, t).sample  # Predicted noise; .sample, because we use diffusers unet
            x_t = self.sample_reverse_process(x_t, t, e_t)
            images.append(x_t)
        
        return images

    def sample_images(self, T: int, batch_size=1, image_shape: tuple = (3, 32, 32), checkpoint_path=None) -> torch.Tensor:
        """
        Returns batch of images after denoising process
        """

        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        tensor = torch.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        x_t = self.noise_like(tensor).to(self.device)

        for t in range(0, T)[::-1]:
            t = torch.full((1,), t).to(self.device)
            e_t = self.model(x_t, t).sample  # Predicted noise; .sample, because we use diffusers unet
            x_t = self.sample_reverse_process(x_t, t, e_t)
        
        return x_t
    