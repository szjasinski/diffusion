from datetime import datetime
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from diffusion import Diffusion
from parameters import RunConfig
from utils.visualization_utils import visualize_process, visualize_grid


def load_transformed_CIFAR10(batch_size: int,
                             seed: int,
                             img_size: int = 32,
                             train: bool = True,
                             path: str  = "./data/") -> Tuple[Dataset, DataLoader]:
    
    """
    Loads transformed CIFAR-10.

    Args:
        batch_size (int): The number of samples per batch in the DataLoader.
        img_size (int, optional): The target size to resize images to (height, width). Defaults to 32.
        train (bool, optional): Whether to load the training set (True) or test set (False). Defaults to True.
        path (str, optional): The root directory for downloading/storing the dataset. Defaults to "./data/".

    Returns:
        Tuple[Dataset, DataLoader]: 
    """
    
    transformation_list = [
        transforms.Resize((img_size, img_size)),  # Resize to specified img_size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]

    data_transform = transforms.Compose(transformation_list)
    dataset = torchvision.datasets.CIFAR10(path, download=True, train=train, transform=data_transform)

    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)

    return dataset, dataloader


def train_model(run_config: RunConfig, dataloader: DataLoader):
    print(f"{datetime.now()} Running experiment {run_config.run_name}...")

    diffuser = Diffusion(run_config.scheduler(run_config.scheduler_T),
                         run_config.noise_func)
    
    print(f"{datetime.now()} Starting training...")

    diffuser.train(dataloader,
                   lr=run_config.training_params.lr,
                   epochs=run_config.training_params.epochs,
                   patience=run_config.training_params.patience,
                   lr_patience=run_config.training_params.lr_patience,
                   factor=run_config.training_params.factor,
                   model_path=run_config.checkpoint_name
                   )
    
    print(f"{datetime.now()} Training finished for {run_config.run_name}.")


def create_visualizations(run_config: RunConfig):

    diffuser = Diffusion(run_config.scheduler(run_config.scheduler_T),
                         run_config.noise_func)

    print(f"{datetime.now()} Visualizing denoising process...")
    backward_process_list = diffuser.get_backward_process_list(T=run_config.scheduler_T, 
                                                               batch_size=run_config.visualization_params.denoising_samples_num, 
                                                               image_shape=run_config.visualization_params.image_shape, 
                                                               model_path=run_config.checkpoint_name)
    visualize_process(backward_process_list, 
                      run_config.visualization_params.n_rows, 
                      run_config.visualization_params.n_cols, 
                      save_result=True, 
                      filename=run_config.run_name)

    print(f"{datetime.now()} Visualizing grid of images...")
    images_batch = diffuser.sample_images(T=run_config.scheduler_T, 
                                          batch_size=run_config.visualization_params.grid_side_size ** 2, 
                                          model_path=run_config.checkpoint_name)
    visualize_grid(images_batch, 
                   save_result=True, 
                   filename=run_config.run_name)
    
    print(f"{datetime.now()} Finished creating visualization for {run_config.run_name}.")
    