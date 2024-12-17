import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Scheduler:
    def __init__(self, num_noise_steps, **kwargs):
        self.num_noise_steps = num_noise_steps
        self.params = kwargs

    def schedule_value_at(self, step):
        raise NotImplementedError

    def generate_schedule(self):
        return np.array([self.schedule_value_at(step) for step in range(self.num_noise_steps)])

class CosineScheduler(Scheduler):
    def schedule_value_at(self, step):
        smoothing_factor = self.params.get('smoothing_factor', 0.008)
        return self._cosine_decay_function(step, smoothing_factor) / self._cosine_decay_function(0, smoothing_factor)

    def _cosine_decay_function(self, step, smoothing_factor):
        return np.cos(
            (((step / self.num_noise_steps) + smoothing_factor) / (1 + smoothing_factor)) * np.pi / 2
        ) ** 2

class LinearScheduler(Scheduler):
    def schedule_value_at(self, step):
        return 1 - step / self.num_noise_steps

class PolynomialScheduler(Scheduler):
    def schedule_value_at(self, step):
        power = self.params.get('power', 2)
        return ((1 - step / self.num_noise_steps) ** power)
    
class ExponentialScheduler(Scheduler):
    def schedule_value_at(self, step):
        scale = self.params.get('scale', 10)
        return np.exp(-scale * (step / self.num_noise_steps))

class InverseScheduler(Scheduler):
    def schedule_value_at(self, step):
        scale = self.params.get('scale', 10)
        return 1 / (1 + scale * (step / self.num_noise_steps))

class LogarithmicScheduler(Scheduler):
    def schedule_value_at(self, step):
        smoothing_factor = self.params.get('smoothing_factor', 0.01)
        return 1 - np.log(1 + smoothing_factor * step) / np.log(1 + smoothing_factor * self.num_noise_steps)

class SigmoidScheduler(Scheduler):
    def schedule_value_at(self, step):
        k = self.params.get('k', 10)  
        b = self.params.get('b', self.num_noise_steps / 2) 
        return 1 - 1 / (1 + np.exp(-k * ((step / self.num_noise_steps) - (b / self.num_noise_steps))))

class NoiseAdder:
    def __init__(self, scheduler_type, num_noise_steps, **scheduler_params):
        self.num_noise_steps = num_noise_steps
        self.scheduler = scheduler_type(num_noise_steps, **scheduler_params)

    def add_noise(self, path):
        image = np.array(Image.open(path).convert("RGB"))
        image = image / 255.0

        noise = np.random.normal(size=image.shape)
        schedule = self.scheduler.generate_schedule()

        noisy_images = []
        for step in range(self.num_noise_steps):
            alpha = schedule[step]
            noisy_image = alpha * image + np.sqrt(1 - alpha) * noise
            noisy_images.append(np.clip(noisy_image, 0, 1))

        return noisy_images

    def get_noisy_image_at_step(self, step, path):
        image = np.array(Image.open(path).convert("RGB"))
        image = image / 255.0

        alpha = self.scheduler.schedule_value_at(step)

        noise = np.random.normal(size=image.shape)
        noisy_image = alpha * image + np.sqrt(1 - alpha) * noise
        noisy_image = np.clip(noisy_image, 0, 1)

        return noisy_image, noise, alpha