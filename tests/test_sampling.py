import importlib
import hydra
import numpy as np
from tqdm import tqdm

from diffusion_lab.models.noise_scheduler import NoiseScheduler
from diffusion_lab.utils.transforms import to_pil
from diffusion_lab.utils.plot_images import show_save_images, save_gif, plot_lines
from diffusion_lab.utils.resolvers import *

from omegaconf import DictConfig

import torch


@torch.no_grad()
def sampling_gif(model, scheduler: NoiseScheduler, x_t, n_images=1, frame_step=1, test_output_path='results/'):
	image_backlog_x_t = [[] for _ in range(n_images)]
	image_backlog_x_0 = [[] for _ in range(n_images)]
	x_0 = x_t
	
	pbar = tqdm(total=scheduler.T - 1)
	for t in reversed(range(1, scheduler.T)):
		# Basic sampling loop
		t_tensor = torch.full((n_images,), t, device=model.device, dtype=torch.long)
		epsilon = model(x_t, t_tensor)
		x_t, x_0 = scheduler.p_backward(x_t, epsilon, t_tensor)
		
		if t % frame_step == 0:
			for i in range(n_images):
				image_backlog_x_t[i].append(to_pil(torch.clamp(x_t[i], -1, 1)))
				image_backlog_x_0[i].append(to_pil(torch.clamp(x_0[i], -1, 1)))
		pbar.update(1)
	pbar.close()
	# Always add last timestep
	for i in range(n_images):
		image_backlog_x_t[i].append(to_pil(torch.clamp(x_t[i], -1, 1)))
		image_backlog_x_0[i].append(to_pil(torch.clamp(x_0[i], -1, 1)))
	
	# Save gifs of all sampled images
	for i in range(n_images):
		save_gif(image_backlog_x_t[i], test_output_path, f"SamplingImage{i}-xt.gif")
		save_gif(image_backlog_x_0[i], test_output_path, f"SamplingImage{i}-x0.gif")


def sampling_image_steps(model, scheduler: NoiseScheduler, x_t, n_images=1, show_steps=None,
                         test_output_path='results/'):
	if show_steps is None:
		show_steps = [1, 500, 999]
	
	line_mean = np.zeros((scheduler.T - 1,))
	line_std = np.zeros((scheduler.T - 1,))
	
	image_backlog = [[] for _ in range(n_images)]
	
	pbar = tqdm(total=scheduler.T - 1)
	for t in reversed(range(1, scheduler.T)):
		# Basic sampling loop
		t_tensor = torch.full((n_images,), t, device=model.device, dtype=torch.long)
		epsilon = model(x_t, t_tensor)
		x_t, x_0 = scheduler.p_backward(x_t, epsilon, t_tensor)
		
		# Some info about images, prep for plotting mean and std of image at timestep
		mean = float(x_t.mean().item())
		line_mean[scheduler.T - t - 1] = mean
		std = float(x_t.std().item())
		line_std[scheduler.T - t - 1] = std
		
		if t in show_steps:
			for i in range(n_images):
				image_backlog[i].append([
					to_pil(torch.clamp(x_t[i], -1, 1)),
					to_pil(torch.clamp(x_0[i], -1, 1)),
				])
		
		pbar.update(1)
	pbar.close()
	
	for i in range(n_images):
		show_save_images(image_backlog[i], test_output_path, f"SampledImage-{i}-timesteps.jpg", show=False)
	
	# Showing Plot of mean and std
	x = np.linspace(scheduler.T - 2, 0, scheduler.T - 1)
	plot_lines(
		x_values=[x, x],
		y_values=[line_mean, line_std],
		line_labels=['Line Mean', 'Line Std'],
		x_label='Diffusion Step',
		y_label='Y-axis',
		title='Mean and Std by Diffusion Step - x_t',
		save_path=test_output_path
	)


@torch.no_grad()
def main(model, scheduler: NoiseScheduler, model_abs_path='steps/final.pth', test_output_path='results/', show_steps=None,
         n_images=1, frame_step=1, resolution=(64, 64), start_noise=None, device='cpu', seed=None, **kwargs):
	model.load_state_dict(torch.load(model_abs_path, map_location=device))
	model.to(device)
	model.eval()
	
	if start_noise is None:
		start_noise = torch.randn((n_images, 3, *resolution)).to(device)
	x_t = start_noise
	
	# Full image sampling animation
	if seed is not None:
		torch.random.manual_seed(seed)
	sampling_gif(model, scheduler, x_t, n_images=n_images, frame_step=frame_step, test_output_path=test_output_path)
	
	# Image sampling with timesteps visible
	if seed is not None:
		torch.random.manual_seed(seed)
	sampling_image_steps(model, scheduler, x_t, n_images=n_images, show_steps=show_steps,
	                     test_output_path=test_output_path)
	

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
