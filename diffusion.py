import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from diffusion_unet import UNet


class Diffusion:

    def __init__(self, T: int = 1000, noise_scheduler: str = "linear"):
        betas_dict = {"linear": self._get_linear_betas,
                      # Add cosine and others
                      }
        if noise_scheduler not in betas_dict:
            raise ValueError(f"Unsupported noise scheduler: {noise_scheduler}")

        betas = betas_dict[noise_scheduler](T=T)

        self.coeffs = self._create_coeffs_from_betas(betas)
        self.T = T

        self.model_path = "model_checkpoint.pth"
        self.model = UNet(T, 3, 3)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        betas = torch.linspace(start, end, T)
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
            noise = torch.randn_like(x_t)
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
        noise = torch.randn_like(x_0)

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
        assert 0 <= t <= self.T
        assert x_t.shape == e_t.shape

        c = self.coeffs
        
        if t > 0:
            sigma_t = c['sqrt_b'][t]   # setting sigma_t**2 to B_t works, variances may be learned by nn.
        else:
            sigma_t = 0
        
        u_t = c['image_coeff'][t] * (x_t - c['noise_coeff'][t] * e_t)
        new_x = u_t + sigma_t * torch.randn_like(x_t)   # sample from p_theta(x_t-1 | x_t)

        return new_x


    def get_loss(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform forward process, predict noise, get MSE
        """

        x_noisy, noise = self.sample_forward_process(x_0, t)
        noise_pred = self.model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)

        return loss


    def train(self, dataloader: DataLoader, lr: float, epochs: int, patience: int = 10, lr_patience: int = 5, factor: float = 0.5) -> None:
        """
        Train UNet model.

        Args:
            dataloader: PyTorch DataLoader providing training data.
            lr: Learning rate.
            epochs: Maximum number of epochs.
            patience: Number of epochs to wait for loss improvement before stopping.
            lr_patience: Number of epochs without improvement before reducing LR.
            factor: Factor to multiply LR when loss plateaus (e.g., 0.5 reduces LR by half).
        """

        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=lr_patience)

        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for (x, _) in dataloader:
                optimizer.zero_grad()
                t = torch.randint(low=0, high=self.T, size=(x.shape[0],))
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
                torch.save(self.model.state_dict(), self.model_path)
                print("New min loss. Model saved.")
            else:
                no_improve_count += 1
            
            # Early stopping check
            if no_improve_count >= patience:
                print(f"Early stopping after {epoch+1}. No improvement for {patience} epochs.")
                break


    
    @torch.no_grad()
    def infer(self, T: int, batch_size=1, image_shape: tuple = (3, 32, 32)) -> list[torch.Tensor]:
        """
        Generate image from noise
        (Algorithm 2 in DDPM paper)
        """

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        x_t = torch.randn((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        images = [x_t]

        for t in range(0, T)[::-1]:
            t = torch.full((1,), t, device=self.device)
            e_t = self.model(x_t, t)  # Predicted noise
            x_t = self.sample_reverse_process(x_t, t, e_t)
            images.append(x_t)
        
        return images
