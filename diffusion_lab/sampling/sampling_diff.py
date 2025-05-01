import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
import hydra
from omegaconf import DictConfig
import importlib
from os.path import join as pjoin
from diffusion_lab.models.noise_scheduler import LinearNoiseScheduler

'''
Sampling process = "Hey, if you saw this noisy image, how would you reverse it?"

based on 'Algorithm 2 Sampling' from DDPM paper
xT~ N(0,I)
for t=T... 1 do
	z~N(0,I) if t>1, else z=0
	xt-1 = 1/sqrt(alpha_t) * ((xt - 1-alpha_t)/(sqrt(1-alpha_bar_t)epislon_theta(xt,t)) + sigmat*z
return x0
'''



@torch.no_grad()
def sample_timestep(model, x, t, scheduler):
    """
    Denoises the image at a single step. It takes the noisy image x and uses
    our model to predict a cleaner version (x_0_pred) then using scheduler it figures out
    how much noise to remove and updates the image.

    Args:
        model: Diffusion model
        x: Current image (x_t), shape (batch_size, channels, height, width)
        t: Timestep (integer)
        scheduler: Noise scheduler instance (reverse_diffusion_step method)

    Returns:
        x_prev: Next image (x_{t-1})
        x_0_pred: Predicted denoised image (x_0)
    """
    
    # Make sure t is a tensor matching with batch size
    if isinstance(t, int):
        t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
    else:
        t_tensor = t

    predicted_noise = model(x, t_tensor)
    x_prev, x_0_pred = scheduler.reverse_diffusion_step(x, predicted_noise, t)
    return x_prev, x_0_pred


@torch.no_grad()
def sample_plots(model, scheduler, image_size, n_samples, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create random noise as start point
    x = torch.randn((n_samples, 3, image_size, image_size), device=device)
    samples = []
    
    # Start from the last timestep T (full noise) and go backwards to 0 (clean image)
    for t in reversed(range(scheduler.T)):
        x, x_0_pred = sample_timestep(model, x, t, scheduler)
        # Save a copy of current prediction every 100 steps or at the first step
        if t % 100 == 0 or t == scheduler.T - 1:
            samples.append(x_0_pred.cpu())
    
    # Create and save grid of final denoised samples
    grid = vutils.make_grid(samples[-1], nrow=int(np.sqrt(n_samples)), normalize=True, value_range=(-1, 1))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Sampled Images after {scheduler.T} Steps")
    plt.savefig(os.path.join(save_dir, "sampled_images.png"))
    plt.close()
    
@hydra.main(config_path="../../config/", config_name="sampler", version_base="1.3")
def main(cfg: DictConfig):
    device = cfg.sample.device

    # Load model
    model_path, model_name = cfg.model.type.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(model_path), model_name)
    model = model_cls(cfg.model.params).to(device)

    # Load checkpoint
    ckpt_path = pjoin("uncond_unet.pth")
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load scheduler (for now it's LinearNoiseScheduler) todo change to: cosine
    scheduler = LinearNoiseScheduler(
        n_timesteps=cfg.scheduler.params.get("n_timesteps", 1000),
        beta_start=cfg.scheduler.params.get("beta_start", 0.001),
        beta_end=cfg.scheduler.params.get("beta_end", 0.02),
        device=device
    )
    
    # Generate and save samples
    sample_plots(model, scheduler, cfg.sample.image_size, cfg.sample.n_samples, cfg.sample.save_dir, device)

if __name__ == "__main__":
    main()
