from pathlib import Path
from typing import Callable
from dataclasses import dataclass, field

from utils.custom_unets.unet_no_timesteps import UNetNoTimesteps
from utils.custom_unets.unet_timesteps import UNetTimesteps
from diffusers import UNet2DModel as UNetDiffusers

from utils.schedulers import  Scheduler


@dataclass
class TrainingParams:
    batch_size: int = 50
    lr: float = 0.0001
    epochs: int = 200    # 1 for testing
    patience: int = 10
    lr_patience: int = 5
    factor: float = 0.5


@dataclass
class VisualizationParams:
    image_shape: tuple[int, int, int] = (3, 32, 32)
    denoising_samples_num: int  = 15  # 2 for testing
    n_rows: int = denoising_samples_num
    n_cols: int = 25
    grid_side_size: int = 6  # 2 for testing


@dataclass
class RunConfig:
    training_params: TrainingParams
    visualization_params: VisualizationParams
    unet_cls: UNetNoTimesteps | UNetTimesteps | UNetDiffusers
    noise_func: Callable
    scheduler: Scheduler
    scheduler_T: int = 1000
    seed: int = 42
    output_folder: str = "outputs"
    experiment_path: str = field(init=False)
    checkpoint_path: str = field(init=False)
    run_name: str = field(init=False)

    def __post_init__(self):
        scheduler_type = self.scheduler.__name__.replace("Scheduler", "").lower()
        noise_type = self.noise_func.__name__.replace("_noise_like", "")
        unet_name = self.unet_cls.__name__.lower()
        self.run_name =  scheduler_type + "_" + noise_type + "_" + unet_name
        self.experiment_path = Path(self.output_folder, self.run_name)
        self.checkpoint_path = Path(self.experiment_path, "checkpoint_" + self.run_name)
