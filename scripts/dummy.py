import torch
import os
from diffusion_lab.models.diffusion import UNet
from diffusion_lab.models.noise_scheduler import LinearNoiseScheduler
from diffusion_lab.sampling.sampling_diff import sample_plots
# 1. Set device
device = "cpu"

# 2. Set your paths
save_dir = "../../real_samples"  # where to save images
os.makedirs(save_dir, exist_ok=True)

# 3. Define model config manually
from omegaconf import DictConfig

cfg_model = DictConfig({
    'base_channels': 64,
    'image_size': 128,
    'in_channels': 3,
    'num_groups': 32,
    'time_embedding_factor': 4,
    'attention_heads': 4,
    'out_channels': 3,
    'num_layers': 2
})

# 4. Load your model
model = UNet(cfg_model, device=device).to(device)
model.load_state_dict(torch.load("../diffusion_lab/sampling/uncond_unet.pth", map_location=device))  # loading trained weights
model.eval()

# 5. Load your scheduler
scheduler = LinearNoiseScheduler(
    n_timesteps=1000,
    beta_start=0.001,
    beta_end=0.02,
    device=device
)

# 6. Call sample_plots
sample_plots(
    model=model,
    scheduler=scheduler,
    image_size=64,       # 128x128 resolution
    n_samples=1,         # generate 1 sample
    save_dir=save_dir,
    device=device
)
