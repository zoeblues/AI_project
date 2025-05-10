import torch

from diffusion_lab.models.noise_scheduler import NoiseScheduler


@torch.no_grad()
def sample_image(model, scheduler: NoiseScheduler, n_timesteps=1_000, n_images=1, resolution=(64, 64), x_T=None):
	model.eval()
	if x_T is None:
		x_t = torch.randn((n_images, 3, *resolution), device=model.device)  # B, C, W, H
	else:
		x_t = x_T
	
	with torch.no_grad():
		for t in reversed(range(1, n_timesteps)):
			t_tensor = torch.full((n_images,), t, device=model.device, dtype=torch.long)
			epsilon = model(x_t, t_tensor)
			x_t, _ = scheduler.p_backward(x_t, epsilon, t_tensor)
			# x_t = torch.clamp(x_t, -3, 3)
	return x_t
