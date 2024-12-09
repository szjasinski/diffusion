import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np

from cosine_noise_adder import CosineNoiseAdder
from unet import UNet


image_path = "goldfish.JPEG"
model_path = "model_checkpoint.pth"

num_noise_steps = 100
num_epochs = 1000
lr = 1e-3

model = UNet(3, 3)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

noiser = CosineNoiseAdder(image_path=image_path, num_noise_steps=num_noise_steps)


for epoch in range(num_epochs):
    t = np.random.randint(1, num_noise_steps)
    noisy_image, noise, alpha = noiser.get_noisy_image_at_step(t)

    noisy_image = torch.from_numpy(noisy_image).float()
    noisy_image = noisy_image.unsqueeze(0)  # Add batch dimension: (1, height, width, channels)
    noisy_image = noisy_image.permute(0, 3, 1, 2) # (batch_size, channels, height, width) for tensors

    noise = torch.from_numpy(noise).float()
    noise = noise.unsqueeze(0)  # Add batch dimension: (1, height, width, channels)
    noise = noise.permute(0, 3, 1, 2) # (batch_size, channels, height, width) for tensors

    optimizer.zero_grad()
    pred_noise = model(noisy_image) # pred_noise = model(noised_image, t) # time information should be used 

    loss = criterion(pred_noise, noise)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
