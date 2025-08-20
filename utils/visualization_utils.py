import os
from datetime import datetime
from typing import Sequence

import torch
import matplotlib.pyplot as plt


def visualize_process(images: Sequence[torch.Tensor], 
                     n_rows: int, 
                     n_cols: int,
                     save_result: bool = False,
                     filename: str = "") -> None:
    """
    Visualizes images at different levels of noise in a grid.

    Args:
        images (Sequence[torch.Tensor]): A sequence where each element is a batch of noised images (Tensor of shape [batch_size, C, H, W]).
        n_rows (int): Number of rows in the grid (batch size).
        n_cols (int): Number of columns in the grid (number of noise levels visualized).
        save_result (bool): Whether to save the resulting grid
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

    if save_result:
        save_dir = "visualizations"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/process-{filename}-{datetime.now()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Image saved: {filename}")

    plt.show()


def visualize_grid(x_batch: torch.Tensor, 
                   save_result: bool = False,
                   filename: str = "") -> None:
    
    batch_size = x_batch.size()[0]
    side_size = batch_size**(1/2)
    assert side_size.is_integer(), "batch size should be a squared integer"
    side_size = int(side_size)

    fig, axes = plt.subplots(side_size, side_size, figsize=(side_size, side_size))
    axes = axes if side_size > 1 else [axes]   # Ensure axes is 2D

    for i in range(batch_size):
        col = i // side_size
        row = i % side_size

        img = x_batch[i].permute(1, 2, 0).cpu().numpy()
        img = img * 0.5 + 0.5  # Denormalize
        img = img.clip(0, 1)
        
        axes[row][col].imshow(img)
        axes[row][col].axis("off")
    
    plt.tight_layout()

    if save_result:
        save_dir = "visualizations"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/grid-{filename}-{datetime.now()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Image saved: {filename}")
    
    plt.show()

