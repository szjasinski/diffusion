from typing import Callable
from dataclasses import dataclass, field

from schedulers import  Scheduler


@dataclass
class TrainingParams:
    batch_size: int = 50
    lr: float = 0.0001
    epochs: int = 120    # 1 for testing
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
    noise_func: Callable
    scheduler: Scheduler
    scheduler_T: int = 1000
    seed: int = 42
    run_name: str = field(init=False)
    checkpoint_name: str = field(init=False)

    def __post_init__(self):
        scheduler_type = self.scheduler.__name__.replace("Scheduler", "").lower()
        noise_type = self.noise_func.__name__.replace("_noise_like", "")
        self.run_name =  scheduler_type + "_" + noise_type
        self.checkpoint_name = "checkpoint_" + self.run_name
