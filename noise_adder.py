import torch
from PIL import Image


class Scheduler:
    def __init__(self, T, **kwargs):
        self.T = T
        self.params = kwargs
        self._compute_coeffs()

    def get_beta(self, t):
        raise NotImplementedError

    def _compute_coeffs(self):
        self.betas = torch.tensor([self.get_beta(t) for t in range(self.T)], dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
        self.sqrt_betas = torch.sqrt(self.betas)
        self.image_coeff = torch.sqrt(1 / self.alphas)
        self.noise_coeff = (1 - self.alphas) / self.sqrt_one_minus_alphas_bar

    def add_noise(self, image: torch.Tensor):
        """Applies noise to an image tensor over the full diffusion process."""
        noisy_images = [image]  

        for t in range(self.T):
            noise = torch.randn_like(noisy_images[-1])
            noisy_image = torch.sqrt(1 - self.betas[t]) * noisy_images[-1] + torch.sqrt(self.betas[t]) * noise
            noisy_images.append(torch.clamp(noisy_image, 0, 1))
    
        return noisy_images

    def get_noisy_image_at_t(self, image: torch.Tensor, t: torch.Tensor):
        """Applies noise to a batch of images at specific diffusion steps."""
        noise = torch.randn_like(image)
        
        sqrt_1m_betas = torch.sqrt(1 - self.betas[t]).view(-1, 1, 1, 1)
        sqrt_betas = torch.sqrt(self.betas[t]).view(-1, 1, 1, 1)

        noisy_image = sqrt_1m_betas * image + sqrt_betas * noise
        return torch.clamp(noisy_image, 0, 1), noise, self.alphas_bar[t]



class CosineScheduler(Scheduler):
    def __init__(self, T, **kwargs):
        """
        Implements a cosine noise schedule with optional smoothing.

        Args:
            T (int): Total number of diffusion steps.
            s (float, optional): Smoothing factor to avoid small alphas at the start. Default is 0.008.
        """
        self.T = T
        self.s = kwargs.get('s', 0.008)  # Smoothing factor

        cos_values = torch.tensor([
            self._cosine_decay_function(torch.tensor(t, dtype=torch.float32))
            for t in range(self.T + 1)
        ], dtype=torch.float32)

        alphas_bar = cos_values / cos_values[0]
        self.betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        self.alphas = 1 - self.betas
        self.alphas_bar = alphas_bar[1:]

        super().__init__(T, **kwargs)

    def get_beta(self, t):
        return self.betas[t]  

    def _cosine_decay_function(self, t):
        """Computes alpha_bar using a cosine decay schedule with smoothing factor `s`."""
        return torch.cos(((t / self.T + self.s) / (1 + self.s)) * torch.pi / 2) ** 2



class LinearScheduler(Scheduler):
    def __init__(self, T, start=0.0001, end=0.02, **kwargs):
        self.betas = torch.linspace(start, end, T, dtype=torch.float32) 
        super().__init__(T, **kwargs)

    def get_beta(self, t):
        return self.betas[t]


class PolynomialScheduler(Scheduler):
    def get_beta(self, t):
        power = self.params.get('power', 2)
        t = torch.tensor(t, dtype=torch.float32)
        return 1 - (t / self.T) ** power
    
class ExponentialScheduler(Scheduler):
    def get_beta(self, t):
        scale = self.params.get('scale', 10)
        t = torch.tensor(t, dtype=torch.float32)
        return 1 - torch.exp(-scale * (t / self.T))

class InverseScheduler(Scheduler):
    def get_beta(self, t):
        scale = self.params.get('scale', 10)
        t = torch.tensor(t, dtype=torch.float32)
        return 1 - (1 / (1 + scale * (t / self.T)))

class LogarithmicScheduler(Scheduler):
    def get_beta(self, t):
        s = self.params.get('s', 0.01)  # Smoothing factor
        t = torch.tensor(t, dtype=torch.float32)
        T_t = torch.tensor(self.T, dtype=torch.float32)
        return 1 - (torch.log(1 + s * t) / torch.log(1 + s * T_t))


class SigmoidScheduler(Scheduler):
    def get_beta(self, t):
        k = self.params.get('k', 10)  
        b = self.params.get('b', self.T / 2) 
        t = torch.tensor(t, dtype=torch.float32)
        return 1 - (1 / (1 + torch.exp(-k * ((t / self.T) - (b / self.T)))))
