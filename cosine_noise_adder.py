import numpy as np
from PIL import Image

class CosineNoiseAdder:
    def __init__(self, image_path, num_noise_steps, smoothing_factor=0.008):
        self.image_path = image_path
        self.num_noise_steps = num_noise_steps
        self.smoothing_factor = smoothing_factor

    def cosine_schedule_value_at(self, step):
        return self.cosine_decay_function(step) / self.cosine_decay_function(0)

    def cosine_decay_function(self, step):
        return np.cos(
            (((step / self.num_noise_steps) + self.smoothing_factor) / (1 + self.smoothing_factor))
            * np.pi / 2
        ) ** 2

    def generate_cosine_schedule(self):
        return np.array([self.cosine_schedule_value_at(step) for step in range(self.num_noise_steps)])

    def add_noise(self):
        image = np.array(Image.open(self.image_path).convert("RGB"))
        image = image / 255.0

        noise = np.random.normal(size=image.shape)
        cosine_schedule = self.generate_cosine_schedule()

        noisy_images = []
        for step in range(self.num_noise_steps):
            alpha = cosine_schedule[step]
            noisy_image = alpha * image + (1 - alpha) * noise
            noisy_images.append(np.clip(noisy_image, 0, 1))

        return noisy_images
