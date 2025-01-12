import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion_unet import UNet
from diffusion import Diffusion


@pytest.fixture
def diffusion():
    """
    Pytest fixture that creates a Diffusion instance with smaller T to speed up tests.
    """
    return Diffusion(T=10, noise_scheduler="linear")


def test_init(diffusion):
    """
    Test that Diffusion initializes without error and that attributes have expected shapes.
    """
    assert diffusion.T == 10
    assert hasattr(diffusion, "coeffs"), "Diffusion should have 'coeffs' attribute."
    assert hasattr(diffusion, "model"), "Diffusion should have 'model' attribute."
    assert isinstance(diffusion.model, UNet), "model should be an instance of UNet."
    assert diffusion.model_path.endswith(".pth")
    
    coeffs = diffusion.coeffs
    required_keys = [
        "sqrt_one_minus_b",
        "sqrt_b",
        "sqrt_a_bar",
        "sqrt_one_minus_a_bar",
        "image_coeff",
        "noise_coeff",
    ]
    for key in required_keys:
        assert key in coeffs, f"Missing {key} in coefficients dictionary."
        assert coeffs[key].shape[0] == diffusion.T, f"{key} should have length T."


def test_create_coeffs_from_betas(diffusion):
    """
    Test private method _create_coeffs_from_betas to ensure it correctly computes coefficients.
    """
    T_test = 5
    betas_test = torch.linspace(0.0001, 0.02, T_test)
    coeffs = diffusion._create_coeffs_from_betas(betas_test)

    for k, v in coeffs.items():
        assert v.shape[0] == T_test, f"Coefficient {k} should have shape [T_test]."


def test_get_linear_betas(diffusion):
    """
    Test _get_linear_betas to ensure it creates the correct beta schedule.
    """
    T_test = 5
    betas = diffusion._get_linear_betas(T=T_test, start=0.1, end=0.2)
    assert betas.shape == (T_test,), "Betas should be a 1D tensor of length T_test."
    assert torch.isclose(betas[0], torch.tensor(0.1), atol=1e-5), "Start value mismatch."
    assert torch.isclose(betas[-1], torch.tensor(0.2), atol=1e-5), "End value mismatch."
    # Ensure it's strictly increasing
    assert torch.all(betas[1:] > betas[:-1]), "Betas should be strictly increasing."


def test_get_forward_process_list(diffusion):
    """
    Test that get_forward_process_list returns T+1 images and correct shapes.
    """
    # Create a dummy input image batch, e.g. (batch_size=2, channels=3, 8x8).
    x_0 = torch.randn(2, 3, 32, 32)
    images = diffusion.get_forward_process_list(x_0)
    
    # Should return T+1 images for each time step from 0..T
    assert len(images) == diffusion.T + 1, "Should return T+1 images."
    
    # Each image should match the shape of x_0
    for img in images:
        assert img.shape == x_0.shape, "Forward process images have incorrect shape."


def test_sample_forward_process(diffusion):
    """
    Test sample_forward_process for shape correctness and valid timesteps.
    """
    x_0 = torch.randn(2, 3, 32, 32)
    t = torch.tensor([3, 5])  # valid time steps within [0, T]

    x_t, noise = diffusion.sample_forward_process(x_0, t)

    # Shapes must match x_0
    assert x_t.shape == x_0.shape, "Noisy image should match original shape."
    assert noise.shape == x_0.shape, "Noise should match original shape."


def test_sample_forward_process_boundary(diffusion):
    """
    Test boundary conditions in sample_forward_process with t=0 and t=T.
    """
    x_0 = torch.randn(2, 3, 32, 32)
    t_min = torch.tensor([0, 0])         # min boundary
    t_max = torch.tensor([diffusion.T-1, diffusion.T-1])  # max boundary

    # Should not raise assertions
    x_t_min, noise_min = diffusion.sample_forward_process(x_0, t_min)
    x_t_max, noise_max = diffusion.sample_forward_process(x_0, t_max)
    
    assert x_t_min.shape == x_0.shape
    assert noise_min.shape == x_0.shape
    assert x_t_max.shape == x_0.shape
    assert noise_max.shape == x_0.shape


def test_sample_reverse_process(diffusion):
    """
    Test sample_reverse_process for shape correctness and boundary checks.
    """
    shape = (2, 3, 32, 32)
    x_t = torch.randn(shape)
    e_t = torch.randn(shape)

    # Middle timestep
    x_t_minus_1 = diffusion.sample_reverse_process(x_t, t=5, e_t=e_t)
    assert x_t_minus_1.shape == shape

    # Boundary timestep t=0 -> returns u_t without adding noise
    x_t0 = diffusion.sample_reverse_process(x_t, t=0, e_t=e_t)
    assert x_t0.shape == shape


def test_get_loss(diffusion):
    """
    Test get_loss to ensure it returns a scalar.
    """
    x_0 = torch.randn(4, 3, 32, 32)
    loss = diffusion.get_loss(x_0)

    assert isinstance(loss, torch.Tensor), "Loss must be a torch.Tensor."
    assert loss.dim() == 0, "Loss should be a scalar (0-dimensional)."


def test_train(diffusion, tmp_path):
    """
    Test that train runs without errors on a small dataset,
    and that it saves a checkpoint.
    """
    # Create a very small dataset
    x_data = torch.randn(4, 3, 32, 32)
    y_data = torch.zeros(4)  # Dummy targets
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=2)

    # Override model_path to write checkpoint to a test directory
    diffusion.model_path = str(tmp_path / "test_model_checkpoint.pth")

    # Train for 1 epoch
    diffusion.train(dataloader, lr=1e-3, epochs=1)

    # Check that checkpoint was saved
    assert (tmp_path / "test_model_checkpoint.pth").exists(), "Checkpoint not saved."



def test_infer(diffusion, tmp_path):
    """
    Test that infer runs without crashing and returns T+1 images.
    For simplicity, we load the same untrained model as the checkpoint.
    """
    # Save current (untrained) state
    torch.save(diffusion.model.state_dict(), diffusion.model_path)

    T_infer = 5
    batch_size = 2
    img_shape = (3, 32, 32)
    images = diffusion.infer(T=T_infer, batch_size=batch_size, image_shape=img_shape)

    # We expect T_infer + 1 images
    assert len(images) == T_infer + 1, "Infer must return T_infer + 1 images."
    for img in images:
        assert img.shape == (batch_size, *img_shape), "Image shape is incorrect."
