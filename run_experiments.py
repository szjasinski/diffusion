from datetime import datetime
import traceback

import torch

from parameters import TrainingParams, VisualizationParams, RunConfig
from schedulers import LinearScheduler, CosineScheduler
from noise_distributions import normal_noise_like, uniform_noise_like, salt_pepper_noise_like
from utils.experiment_utils import (load_transformed_CIFAR10,
                              train_model,
                              create_visualizations,)



train_params = TrainingParams()
vis_params = VisualizationParams()
 
runs = {
    0: RunConfig(train_params, vis_params, normal_noise_like, LinearScheduler),
    1: RunConfig(train_params, vis_params, normal_noise_like, CosineScheduler),
    2: RunConfig(train_params, vis_params, uniform_noise_like, CosineScheduler),
    3: RunConfig(train_params, vis_params, salt_pepper_noise_like, CosineScheduler),
}

# Select indexes and order of configs to run
indexes = [2, 1, 0]


if __name__ == "__main__":
    seed = RunConfig.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"{datetime.now()} Loading dataset...")
    data, dataloader = load_transformed_CIFAR10(batch_size=TrainingParams.batch_size, seed=seed)

    for i in indexes:
        run_config = runs[i]
        try:
            train_model(run_config, dataloader)
            create_visualizations(run_config)
        except Exception as e:
            print(f"{datetime.now()} Run {run_config.run_name} failed.")
            print("\n** ERROR **", e)
            traceback.print_exc()
