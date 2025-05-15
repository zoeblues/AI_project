import importlib

import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusion_lab.models.noise_scheduler import CosineNoiseScheduler, LinearNoiseScheduler, NoiseScheduler
# from diffusion_lab.models.diffusion import UNet
from diffusion_lab.models.unet import UNet
from diffusion_lab.sampling.sampling import sample_image
from diffusion_lab.utils.transforms import to_pil

from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import transforms
from PIL import Image

import torch


@torch.no_grad()
def main(model, scheduler: NoiseScheduler, model_abs_path='steps/final.pth', show_steps=None, n_images=1, resolution=(64, 64), start_noise=None, device='cpu', **kwargs):
	if show_steps is None:
		show_steps = [1, 500, 999]
	
	model.load_state_dict(torch.load(model_abs_path, map_location=device))
	model.to(device)
	model.eval()
	
	if start_noise is None:
		start_noise = torch.randn((n_images, 3, *resolution), device=model.device)  # B, C, W, H
	x_t = start_noise
	
	line_mean = np.zeros((scheduler.T-1,), dtype=float)
	line_std = np.zeros((scheduler.T-1,))
	
	n_columns = len(show_steps)
	bgc = Image.new("RGB", (64 * n_columns, 64 * 2), color=(255, 255, 255)).convert("RGB")
	pbar = tqdm(total=scheduler.T - 1)
	for t in reversed(range(1, scheduler.T)):
		t_tensor = torch.full((n_images,), t, device=model.device, dtype=torch.long)
		epsilon = model(x_t, t_tensor)
		x_t, x_0 = scheduler.p_backward(x_t, epsilon, t_tensor, epsilon_t=start_noise)
		# test = x_t.detach().cpu().numpy()
		
		# Some info
		mean = float(x_t.mean().item())
		std = float(x_t.std().item())
		# print(f"Step {t}: mean={mean:.3f}, std={std:.3f}")
		line_mean[scheduler.T - t - 1] = mean
		line_std[scheduler.T - t - 1] = std
		
		if t in show_steps:
			bgc.paste(to_pil(x_t[0]), (64 * show_steps.index(t), 0))
			bgc.paste(to_pil(x_0[0]), (64 * show_steps.index(t), 64))
		
		pbar.update(1)
	pbar.close()
	
	x = np.linspace(scheduler.T-2, 0, scheduler.T-1)
	plt.plot(x, line_mean, label='Line Mean')
	plt.plot(x, line_std, label='Line Std')
	
	plt.xlabel('Diffusion Step')
	plt.ylabel('Y-axis')
	plt.title('Mean and Std by Diffusion Step')
	
	plt.legend()
	plt.show()
	
	bgc.show()


@hydra.main(config_path="../config", config_name="diffusion", version_base="1.3")
def load_run(cfg: DictConfig):
	model_path, model_name = cfg.model.type.rsplit(".", maxsplit=1)
	model_cls = getattr(importlib.import_module(model_path), model_name)
	model = model_cls(**cfg.model.params)
	
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler_cls = getattr(importlib.import_module(sche_path), sche_name)
	scheduler = scheduler_cls(**cfg.schedule.params)
	
	main(model, scheduler, resolution=(cfg.run.img_size, cfg.run.img_size), **cfg.tests.params)


if __name__ == '__main__':
	load_run()
