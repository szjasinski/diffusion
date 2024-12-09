import torch
import numpy as np

from PIL import Image

from unet import UNet
from cosine_noise_adder import CosineNoiseAdder

model_path = "model_checkpoint.pth"
image_path = "goldfish.JPEG" # needed only for defining noiser, isnt used

num_noise_steps = 13 # needed for alphas

model = UNet(3, 3)

model.load_state_dict(torch.load(model_path))
model.eval()


noise = torch.normal(0, 1, size=(3, 512, 512)).unsqueeze(0)
noiser = CosineNoiseAdder(image_path=image_path, num_noise_steps=num_noise_steps)
alphas = noiser.generate_cosine_schedule()[::-1]

for step, alpha in zip(range(num_noise_steps), alphas):
    print(step)

    pred_noise = model(noise)
    denoised_image = (noise - pred_noise) / alpha + pred_noise
    noise = denoised_image


image_numpy = noise.detach().numpy().squeeze()  # Squeeze to remove batch dimension (1, 3, 128, 128) -> (3, 128, 128)
image_numpy *= 255
image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
image_numpy = np.transpose(image_numpy, (1, 2, 0))  # (3, 128, 128) -> (128, 128, 3)
image = Image.fromarray(image_numpy)
image.show()

