import torch

from diffusion import Diffusion
from noise_adder import LinearScheduler, CosineScheduler
from utils import visualize_process, visualize_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


T = 1000
image_shape = (3, 32, 32)
batch_size = 15
n_rows = batch_size
n_cols = 25

grid_side_size = 6

schedulers = [LinearScheduler(T=T), CosineScheduler(T=T)]
checkpoints = ['checkpoint_linear', "checkpoint_cosine"]
filenames = ['linear', 'cosine']


for sched, checkpoint, filename in zip(schedulers, checkpoints, filenames):

    diffuser = Diffusion(scheduler=sched)

    if diffuser.scheduler.__class__.__name__ == "CosineScheduler":
        diffuser.coeffs['image_coeff'][-1] = 1.9999

    backward_process_list = diffuser.get_backward_process_list(T=T, batch_size=batch_size, image_shape=image_shape, model_path=checkpoint)
    visualize_process(backward_process_list, n_rows, n_cols, save_result=True, filename=filename)

    images_batch = diffuser.sample_images(T=T, batch_size=grid_side_size**2, model_path=checkpoint)
    visualize_grid(images_batch, save_result=True, filename=filename)

