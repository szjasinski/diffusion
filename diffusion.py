import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader

from diffusion_unet import UNet


class Diffusion:

    def __init__(self):
        self.schedulers = dict()
        self.schedulers['linear'] = self.create_linear_scheduler(T=1000, start=0.0001, end=0.02)

        self.model_path = "model_checkpoint.pth"
        self.model = UNet(3, 3)


    def create_linear_scheduler(self, T: int, start: float, end: float) -> dict:
        # DDPM paper defaults: T=1000, start=0.0001, end=0.02
        betas = torch.linspace(start, end, T)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        linear_scheduler = dict()
        linear_scheduler["betas"] = betas
        linear_scheduler["alphas"] = alphas
        linear_scheduler["alphas_bar"] = alphas_bar

        return linear_scheduler


    def q_iteratively(self, x_0: torch.Tensor, t: int, scheduler: str = 'linear') -> list[torch.Tensor]:
        """
        Samples a new image from q step by step
        May be used to visualize forward process later
        Returns the noised images
        x_0: the original image
        t: timestep (t equal to scheduler length will perform complete forward diffusion)
        scheduler: scheduler 
        """

        B = self.schedulers[scheduler]['betas']
        x_t = x_0
        images = [x_t]

        # sample step by step 
        for i in range(t):
            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(1 - B[i]) * x_t + torch.sqrt(B[i]) * noise  # sample from q(x_t|x_t-1)
            images.append(x_t)
        
        return images


    def q(self, x_0: torch.Tensor, t: torch.Tensor, scheduler: str = 'linear') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a new image from q in one step
        Used for training the model
        Returns the noised image and the noise applied to an image at timestep t
        x_0: the original image
        t: timesteps tensor (different times for different images)
        scheduler: scheduler 
        """

        a_bar = self.schedulers[scheduler]["alphas_bar"]
        noise = torch.randn_like(x_0)

        # sample in one step
        x_t = torch.sqrt(a_bar)[t] * x_0 + torch.sqrt(1 - a_bar)[t] * noise # sample from q(x_t | x_0)

        return x_t, noise


    @torch.no_grad()
    def q_reverse(self, x_t: torch.Tensor, t: int, e_t: torch.Tensor, scheduler: str = 'linear') -> torch.Tensor:
        """
        Samples a new image from reverse q (one iteration of Algorithm 2 in DDPM paper)
        Returns x_(t-1) 
        x_t: noised image
        t: timestep
        e_t: predicted noise
        scheduler: scheduler
        """

        B = self.schedulers[scheduler]['betas']
        a = self.schedulers[scheduler]['alphas']
        a_bar = self.schedulers[scheduler]['alphas_bar']

        image_scaling_t = torch.sqrt(1 / a)[t]
        noise_scaling_t = ((1 - a) / torch.sqrt(1 - a_bar))[t]

        u_t = image_scaling_t * (x_t - noise_scaling_t * e_t)   # Estimated mean
        
        if t == 0:
            return u_t
        else:
            B_t = B[t-1]    # why??
            sigma_t = torch.sqrt(B_t)   # setting sigma_t to B_t works, variances may be learned by nn
            z = torch.randn_like(x_t)
            new_x = u_t + sigma_t * z
            return new_x


    def get_loss(self, model: nn.Module, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_noisy, noise = self.q(x_0, t)
        noise_pred = model(x_noisy)  # model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)
        # print(loss.shape)
        return loss


    def train(self, dataloader: DataLoader, T: int, lr: float, epochs: int, batch_size: int) -> None:
        "Train Unet model"

        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                t = torch.randint(low=0, high=T, size=(batch_size, 1, 1, 1)) # t must be 4D tensor
                x = batch[0]
                loss = self.get_loss(model, x, t)
                loss.backward()
                optimizer.step()

                if epoch % 1 == 0 and step % 100 == 0:
                    print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item()} ")
                
            torch.save(model.state_dict(), self.model_path)

    
    @torch.no_grad()
    def infer(self, T: int) -> list[torch.Tensor]:
        """
        Generate image from noise
        (Algorith 2 in DDPM paper)
        """

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        # Noise to generate images from
        x_t = torch.randn((1, 3, 64, 64))
        images = [x_t]

        # Go from T to 0
        for t in range(0, T)[::-1]:
            e_t = self.model(x_t)  # Predicted noise, self.model(x_t, t)
            x_t = self.q_reverse(x_t, t, e_t)
            images.append(x_t)
        
        return images
