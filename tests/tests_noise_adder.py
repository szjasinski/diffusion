import pytest
import numpy as np
from PIL import Image
from io import BytesIO


from diffusion.noise_adder import (
    CosineScheduler,
    LinearScheduler,
    PolynomialScheduler,
    ExponentialScheduler,
    InverseScheduler,
    LogarithmicScheduler,
    SigmoidScheduler,
    NoiseAdder
)

# Test if all schedulers generate valid schedules
# Ensures values are in [0,1] and start from ~1 to ~0
def test_scheduler_values():
    num_steps = 10
    schedulers = [
        CosineScheduler(num_steps),
        LinearScheduler(num_steps),
        PolynomialScheduler(num_steps, power=2),
        ExponentialScheduler(num_steps, scale=10),
        InverseScheduler(num_steps, scale=10),
        LogarithmicScheduler(num_steps, smoothing_factor=0.01),
        SigmoidScheduler(num_steps, k=10, b=num_steps / 2)
    ]
    
    for scheduler in schedulers:
        schedule = scheduler.generate_schedule()
        assert isinstance(schedule, np.ndarray)
        assert schedule.shape == (num_steps,)
        assert np.all(schedule >= 0) and np.all(schedule <= 1)
        assert np.isclose(schedule[0], 1, atol=0.1) 
        assert np.isclose(schedule[-1], 0, atol=0.1) 

# Test if NoiseAdder correctly adds noise to an image
# Ensures noise-added images have the correct shape and valid pixel values
def test_noise_adder():
    num_steps = 5
    image_size = (64, 64, 3)
    dummy_image = (np.random.rand(*image_size) * 255).astype(np.uint8)
    image = Image.fromarray(dummy_image)
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    schedulers = [CosineScheduler, LinearScheduler, PolynomialScheduler]
    
    for scheduler in schedulers:
        noise_adder = NoiseAdder(scheduler, num_steps)
        noisy_images = noise_adder.add_noise(buffer)
        
        assert len(noisy_images) == num_steps
        for noisy_img in noisy_images:
            assert noisy_img.shape == image_size
            assert np.all(noisy_img >= 0) and np.all(noisy_img <= 1)

# Test if NoiseAdder correctly generates a noisy image at a specific step
# Ensures the output shape, noise consistency, and alpha values are valid
def test_get_noisy_image_at_step():
    num_steps = 5
    image_size = (64, 64, 3)
    dummy_image = (np.random.rand(*image_size) * 255).astype(np.uint8)
    image = Image.fromarray(dummy_image)
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    scheduler = LinearScheduler(num_steps)
    noise_adder = NoiseAdder(LinearScheduler, num_steps)
    noisy_image, noise, alpha = noise_adder.get_noisy_image_at_step(2, buffer)
    
    assert noisy_image.shape == image_size
    assert noise.shape == image_size
    assert 0 <= alpha <= 1

if __name__ == "__main__":
    pytest.main([__file__])
