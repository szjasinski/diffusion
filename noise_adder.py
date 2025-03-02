import torch
from PIL import Image


class Scheduler:
    def __init__(self, num_noise_steps, **kwargs):
        self.num_noise_steps = num_noise_steps
        self.params = kwargs
        self.coeffs = self._compute_coeffs()

    def schedule_value_at(self, step):
        raise NotImplementedError

    def _compute_coeffs(self):
        betas = torch.tensor([self.schedule_value_at(step) for step in range(self.num_noise_steps)], dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        coeffs = {
            "betas": betas,
            "alphas": alphas,
            "alphas_bar": alphas_bar,
            "sqrt_alphas_bar": torch.sqrt(alphas_bar),
            "sqrt_one_minus_alphas_bar": torch.sqrt(1 - alphas_bar),
            "sqrt_betas": torch.sqrt(betas),
        }
        return coeffs

    def add_noise(self, image: torch.Tensor):
        """Applies noise to an image tensor over the full diffusion process."""
        c = self.coeffs
        noisy_images = []
        for step in range(self.num_noise_steps):
            noise = torch.randn_like(image)
            noisy_image = c["sqrt_alphas_bar"][step] * image + c["sqrt_one_minus_alphas_bar"][step] * noise
            noisy_images.append(torch.clamp(noisy_image, 0, 1))
        return noisy_images

    def get_noisy_image_at_step(self, image: torch.Tensor, step: int):
        """Applies noise to an image tensor at a specific diffusion step."""
        c = self.coeffs
        noise = torch.randn_like(image)
        noisy_image = c["sqrt_alphas_bar"][step] * image + c["sqrt_one_minus_alphas_bar"][step] * noise
        return torch.clamp(noisy_image, 0, 1), noise, c["alphas_bar"][step]

class CosineScheduler(Scheduler):
    def __init__(self, num_noise_steps, **kwargs):
        self.num_noise_steps = num_noise_steps
        self.smoothing_factor = kwargs.get('smoothing_factor', 0.008) 

        cos_values = torch.tensor([
            self._cosine_decay_function(torch.tensor(t, dtype=torch.float32), self.smoothing_factor)
            for t in range(self.num_noise_steps + 1)
        ], dtype=torch.float32)

        alphas_bar = cos_values / cos_values[0]  
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])  
        alphas = 1 - betas 

        self.alphas_bar = alphas_bar[1:]  
        self.betas = betas
        self.alphas = alphas
        super().__init__(num_noise_steps, **kwargs)

    def schedule_value_at(self, step):
        return self.betas[step]  

    def _cosine_decay_function(self, step, smoothing_factor):
        return torch.cos(
            (((step / self.num_noise_steps) + smoothing_factor) / (1 + smoothing_factor)) * torch.pi / 2
        ) ** 2


class LinearScheduler(Scheduler):
    def __init__(self, num_noise_steps, start=0.0001, end=0.02, **kwargs):
        self.betas = torch.linspace(start, end, num_noise_steps, dtype=torch.float32) 
        super().__init__(num_noise_steps, **kwargs)

    def schedule_value_at(self, step):
        return self.betas[step]


class PolynomialScheduler(Scheduler):
    def schedule_value_at(self, step):
        power = self.params.get('power', 2)
        step_t = torch.tensor(step, dtype=torch.float32)
        return 1 - (step_t / self.num_noise_steps) ** power
    
class ExponentialScheduler(Scheduler):
    def schedule_value_at(self, step):
        scale = self.params.get('scale', 10)
        step_t = torch.tensor(step, dtype=torch.float32)
        return 1 - torch.exp(-scale * (step_t / self.num_noise_steps))

class InverseScheduler(Scheduler):
    def schedule_value_at(self, step):
        scale = self.params.get('scale', 10)
        step_t = torch.tensor(step, dtype=torch.float32)
        return 1 - (1 / (1 + scale * (step_t / self.num_noise_steps)))

class LogarithmicScheduler(Scheduler):
    def schedule_value_at(self, step):
        smoothing_factor = self.params.get('smoothing_factor', 0.01)
        step_t = torch.tensor(step, dtype=torch.float32)
        num_steps_t = torch.tensor(self.num_noise_steps, dtype=torch.float32)
        return 1 - (torch.log(1 + smoothing_factor * step_t) / torch.log(1 + smoothing_factor * num_steps_t))


class SigmoidScheduler(Scheduler):
    def schedule_value_at(self, step):
        k = self.params.get('k', 10)  
        b = self.params.get('b', self.num_noise_steps / 2) 
        step_t = torch.tensor(step, dtype=torch.float32)
        return 1 - (1 / (1 + torch.exp(-k * ((step_t / self.num_noise_steps) - (b / self.num_noise_steps)))))