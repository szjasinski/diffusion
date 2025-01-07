import numpy as np
import pytest
from PIL import Image

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

def test_cosine_scheduler():
    scheduler = CosineScheduler(num_noise_steps=10)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] == 1  
    assert schedule[-1] <= 1  

def test_linear_scheduler():
    scheduler = LinearScheduler(num_noise_steps=10)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] == 1
    assert schedule[-1] <= 1

def test_polynomial_scheduler():
    scheduler = PolynomialScheduler(num_noise_steps=10, power=2)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] == 1
    assert schedule[-1] <= 1

def test_exponential_scheduler():
    scheduler = ExponentialScheduler(num_noise_steps=10, scale=1)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] == 1
    assert schedule[-1] < 1

def test_inverse_scheduler():
    scheduler = InverseScheduler(num_noise_steps=10, scale=1)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] == 1
    assert schedule[-1] < 1

def test_logarithmic_scheduler():
    scheduler = LogarithmicScheduler(num_noise_steps=10, smoothing_factor=0.01)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] == 1
    assert schedule[-1] < 1

def test_sigmoid_scheduler():
    scheduler = SigmoidScheduler(num_noise_steps=10, k=1, b=5)
    schedule = scheduler.generate_schedule()
    assert len(schedule) == 10
    assert schedule[0] < 1
    assert schedule[-1] > 0
