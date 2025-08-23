from datetime import datetime
import traceback
from tqdm import tqdm

import torch

from utils.parameters import TrainingParams, VisualizationParams, RunConfig
from utils.schedulers import LinearScheduler, CosineScheduler
from utils.noise_distributions import normal_noise_like, uniform_noise_like, salt_pepper_noise_like
from utils.experiment_utils import (load_transformed_CIFAR10,
                                    train_model,
                                    create_visualizations,
                                    log_config,)
from utils.custom_unets.unet_no_timesteps import UNetNoTimesteps
from utils.custom_unets.unet_timesteps import UNetTimesteps
from diffusers import UNet2DModel as UNetDiffusers


train_params = TrainingParams()
vis_params = VisualizationParams()
 
runs = {
    0: RunConfig(train_params, vis_params, UNetNoTimesteps, normal_noise_like, LinearScheduler),
    1: RunConfig(train_params, vis_params, UNetNoTimesteps, normal_noise_like, CosineScheduler),
    2: RunConfig(train_params, vis_params, UNetTimesteps, normal_noise_like, LinearScheduler),
    3: RunConfig(train_params, vis_params, UNetTimesteps, normal_noise_like, CosineScheduler),
    4: RunConfig(train_params, vis_params, UNetDiffusers, normal_noise_like, LinearScheduler),
    5: RunConfig(train_params, vis_params, UNetDiffusers, normal_noise_like, CosineScheduler),
    6: RunConfig(train_params, vis_params, UNetDiffusers, uniform_noise_like, CosineScheduler),
    7: RunConfig(train_params, vis_params, UNetDiffusers, salt_pepper_noise_like, CosineScheduler),
}

# Select indexes and order of configs to run
# indexes = [1, 2, 0, 3]
indexes = runs.keys()


if __name__ == "__main__":
    seed = RunConfig.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"{datetime.now()} Loading dataset...")
    data, dataloader = load_transformed_CIFAR10(batch_size=TrainingParams.batch_size, seed=seed)

    for i in tqdm(indexes):
        run_config = runs[i]
        try:
            log_config(run_config)
            train_model(run_config, dataloader)
            create_visualizations(run_config)
        except Exception as e:
            print(f"{datetime.now()} Run {run_config.run_name} failed.")
            print("\n** ERROR **", e)
            traceback.print_exc()
