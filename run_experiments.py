import torch

from diffusion import Diffusion
from noise_adder import LinearScheduler, CosineScheduler
from noise_distributions import normal_noise_like, uniform_noise_like, salt_pepper_noise_like

from utils import load_transformed_CIFAR10
from utils import visualize_process, visualize_grid

from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training params
batch_size = 50
epochs = 120    # 1 for testing
lr = 0.0001
T = 1000
#########


# Visualization params
image_shape = (3, 32, 32)
denoising_samples_num = 15  # 2 for testing
n_rows = denoising_samples_num
n_cols = 25
grid_side_size = 6  # 2 for testing
#########

data, dataloader = load_transformed_CIFAR10(batch_size=batch_size)

runs = {"names": ["cos_normal", "cos_uniform", "cos_salt_pepper"],
        "checkpoints": ["checkpoint_cos_normal", "checkpoint_cos_uniform", "checkpoint_cos_salt_pepper"],   # will be created
        "schedulers": [CosineScheduler(T=T), CosineScheduler(T=T), CosineScheduler(T=T)],
        "noises": [normal_noise_like, uniform_noise_like, salt_pepper_noise_like]}


# TRAIN AND VISUALIZE LOOP
indexes = [2, 1, 0] # Choose order
for i in indexes:
    file_prefix = runs["names"][i]
    checkpoint = runs['names'][i]
    scheduler = runs["schedulers"][i]
    noise_distribution = runs["noises"][i]
    print("RUN", i)
    print(file_prefix)
    print(datetime.now())

    diffuser = Diffusion(scheduler, noise_distribution)

    print("Training...")
    diffuser.train(dataloader, epochs=epochs, lr=lr, model_path=checkpoint) # Comment if only inference is needed
    print(datetime.now())
    

    # VISUALIZATIONS
    if diffuser.scheduler.__class__.__name__ == "CosineScheduler":
        diffuser.coeffs['image_coeff'][-1] = 1.9999 # change inf value to number. happens because we start with betas in diffusion with cosine scheduler and not with alphas

    print("Denoising visualization...")
    backward_process_list = diffuser.get_backward_process_list(T=T, batch_size=denoising_samples_num, image_shape=image_shape, model_path=checkpoint)
    visualize_process(backward_process_list, n_rows, n_cols, save_result=True, filename=file_prefix)

    print("Grid visualization...")
    images_batch = diffuser.sample_images(T=T, batch_size=grid_side_size**2, model_path=checkpoint)
    visualize_grid(images_batch, save_result=True, filename=file_prefix)
