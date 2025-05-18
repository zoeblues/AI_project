import torch
from tqdm import tqdm

from diffusion_lab.models.noise_scheduler import NoiseScheduler

import torch

from diffusion_lab.models.noise_scheduler import NoiseScheduler


@torch.no_grad()
def sample_image(model, scheduler: NoiseScheduler, n_timesteps=1_000, n_images=1, resolution=(64, 64), x_t=None) -> torch.Tensor:
	model.eval()
	if x_t is None:
		x_t = torch.randn((n_images, 3, *resolution), device=model.device)  # B, C, W, H
	
	pbar = tqdm(total=scheduler.T - 1)
	for t in reversed(range(1, n_timesteps)):
		t_tensor = torch.full((n_images,), t, device=model.device, dtype=torch.long)
		epsilon = model(x_t, t_tensor)
		x_t, _ = scheduler.p_backward(x_t, epsilon, t_tensor)
		pbar.update(1)
	pbar.close()
	
	return torch.clamp(x_t, -1, 1)
