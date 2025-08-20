import torch


def normal_noise_like(x):
    return torch.randn_like(x)


def uniform_noise_like(x):
    return (torch.rand_like(x)*2-1) * 3**0.5   # mean 0, var 1


def salt_pepper_noise_like(x):

    b, c, h, w = x.shape
    mask = torch.rand((b, 1, h, w), device=x.device, dtype=x.dtype)   # one random value per pixel
    noise_per_pixel = torch.where(mask < 0.5,
                                  torch.full_like(mask, -1),    # salt
                                  torch.ones_like(mask))    # pepper
    
    return noise_per_pixel.expand(-1, c, -1, -1)    # broadcast