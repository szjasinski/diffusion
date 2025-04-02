import torch

from utils import load_transformed_CIFAR10
from utils import visualize_process

from diffusion import Diffusion
from noise_adder import LinearScheduler, CosineScheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 50
epochs = 120
lr = 0.0001

data, dataloader = load_transformed_CIFAR10(batch_size=batch_size)

model_path="checkpoint_cosine"
scheduler = CosineScheduler(T=1000)
diffuser = Diffusion(scheduler=scheduler)

diffuser.train(dataloader, epochs=epochs, lr=lr, model_path=model_path)


# Visualization
T = diffuser.T
image_shape = (3, 32, 32)
batch_size = 15
n_rows = batch_size
n_cols = 25

backward_process_list = diffuser.get_backward_process_list(T=T, batch_size=batch_size, image_shape=image_shape, model_path=model_path)
visualize_process(backward_process_list, n_rows, n_cols, save_result=True)
