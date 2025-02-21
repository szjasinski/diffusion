import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

from typing import Tuple, Sequence


def load_transformed_CIFAR10_subset(batch_size: int, 
                                    img_size: int = 32,
                                    train: bool = True,
                                    label: int = 1,
                                    path: str  = "./data/") -> Tuple[Subset, DataLoader]:
    
    """
    Loads a subset of the CIFAR-10 dataset containing only images of a specific class label, 
    applies transformations, and returns both a dataset subset and a DataLoader.

    Args:
        batch_size (int): The number of samples per batch in the DataLoader.
        img_size (int, optional): The target size to resize images to (height, width). Defaults to 32.
        train (bool, optional): Whether to load the training set (True) or test set (False). Defaults to True.
        label (int, optional): The class label to filter (e.g., 1 for automobiles). Defaults to 1.
        path (str, optional): The root directory for downloading/storing the dataset. Defaults to "./data/".

    Returns:
        Tuple[Subset, DataLoader]: 
            - A `Subset` of CIFAR-10 containing only images of the specified class label.
            - A `DataLoader` for iterating over the subset.
    """
    
    transformation_list = [
        transforms.Resize((img_size, img_size)),  # Resize to specified img_size
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]

    data_transform = transforms.Compose(transformation_list)
    dataset = torchvision.datasets.CIFAR10(path, download=True, train=train, transform=data_transform)

    subset_indices = [idx for idx, lbl in enumerate(dataset.targets) if lbl == label]
    subset = Subset(dataset, subset_indices)

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)

    return subset, dataloader


def visualize_images(images: Sequence[torch.Tensor], n_rows: int, n_cols: int) -> None:
    """
    Visualizes images at different levels of noise in a grid.

    Args:
        images (Sequence[torch.Tensor]): A sequence where each element is a batch of noised images (Tensor of shape [batch_size, C, H, W]).
        n_rows (int): Number of rows in the grid (batch size).
        n_cols (int): Number of columns in the grid (number of noise levels visualized).
    """

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    axes = axes if n_rows > 1 else [axes]   # Ensure axes is 2D
    step = len(images) // n_cols

    for row in range(n_rows):
        for col in range(n_cols):
            idx = col * step
            img = images[idx][row].permute(1, 2, 0).cpu().numpy()
            img = img * 0.5 + 0.5  # Denormalize
            img = img.clip(0, 1)
            
            axes[row][col].imshow(img)
            axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()

    return None
