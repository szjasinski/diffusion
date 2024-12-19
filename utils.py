import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt


def load_CIFAR10_automobiles(data_transform, max_samples=100, train=True):
    dataset = torchvision.datasets.CIFAR10(
        "./data/",
        download=True,
        train=train,
        transform=data_transform,
    )

    automobiles_indices = [idx for idx, label in enumerate(dataset.targets) if label == 1]
    selected_indices = automobiles_indices[:max_samples]
    subset = Subset(dataset, selected_indices)

    return subset


def load_transformed_CIFAR10_automobiles(batch_size, max_samples=100, img_size=32):
    data_transforms = [
        transforms.Resize((img_size, img_size)),  # Resize to specified img_size
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize for CIFAR-10
    ]

    data_transform = transforms.Compose(data_transforms)
    train_set = load_CIFAR10_automobiles(data_transform, max_samples=max_samples, train=True)
    test_set = load_CIFAR10_automobiles(data_transform, train=False)
    data = ConcatDataset([train_set, test_set])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return data, dataloader


def visualize_images(images: list[torch.Tensor], n_rows: int, n_cols: int) -> None:
    """
    Visualizes images at different levels of noise in a grid.

    Args:
        images (list): A list where each element is a batch of noised images.
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
