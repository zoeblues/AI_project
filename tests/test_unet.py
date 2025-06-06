import importlib
import math

import hydra
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diffusion_lab.models.noise_scheduler import NoiseScheduler
from diffusion_lab.utils.transforms import to_tensor, to_pil
from diffusion_lab.utils.resolvers import *


@torch.no_grad()
def test_denoise(model, scheduler: NoiseScheduler, img_tensor: torch.Tensor, timestep: int, device='cpu'):
	img_tensor = img_tensor.to(device)
	if not isinstance(timestep, torch.Tensor):
		timestep = torch.tensor([timestep], device=device)
	
	# Change to batch form (shape) -> (1, shape)
	img_tensor = img_tensor.unsqueeze(0)
	
	torch.manual_seed(0)
	noise = torch.randn_like(img_tensor, device=device)
	
	noised_img, noise = scheduler.q_forward(img_tensor, timestep, epsilon=noise)
	# img = noise[0][0].cpu().numpy()
	# img = noised_img[0][0].cpu().numpy()
	
	pred = model(noised_img, timestep)
	less_noised_img, clear_img = scheduler.p_backward(noised_img, pred, timestep)
	
	return noised_img[0], noise[0], pred[0], less_noised_img[0], clear_img[0]


def main(model, scheduler, model_abs_path='steps/final.pth', test_img_path='test.jpg', show_steps=None, device='cpu',
         **kwargs):
	if show_steps is None:
		show_steps = [1, 500, 999]
	
	image = Image.open(test_img_path).convert('RGB')
	
	model.load_state_dict(torch.load(model_abs_path, map_location=device))
	model.to(device)
	model.eval()
	
	n_rows = 3
	n_columns = len(show_steps)
	
	bgc = Image.new("RGB", (64 * n_columns, 64 * n_rows), color=(255, 255, 255)).convert("RGB")
	for i, t in enumerate(show_steps):
		noised_img, noise, pred_noise, _, pred_img = test_denoise(model, scheduler, to_tensor(image), t, device=device)
		# img = pred_img[0].cpu().numpy()
		
		loss = torch.nn.MSELoss()(pred_noise, noise).item()
		print(f"Step {t}: loss = {loss:.4f}, mean: {noise.mean():.2f}, std: {noise.std():.2f}")
		
		bgc.paste(to_pil(noised_img), (64 * i, 0))
		bgc.paste(to_pil(noise - pred_noise), (64 * i, 64))
		bgc.paste(to_pil(pred_img), (64 * i, 128))
	
	bgc.show()


@torch.no_grad()
def plot_timestep_loss(model, scheduler, model_abs_path='final.pth', test_img_path='test.jpg', device='cpu', **kwargs):
	model.load_state_dict(torch.load(model_abs_path, map_location=device))
	model.to(device)
	model.eval()
	
	image = Image.open(test_img_path).convert('RGB')
	img_tensor = to_tensor(image).to(device).unsqueeze(0)
	
	torch.manual_seed(0)
	noise = torch.randn_like(img_tensor, device=device)
	
	losses = np.zeros((scheduler.T - 1,))
	pbar = tqdm(total=scheduler.T)
	for t in range(1, scheduler.T):
		t_tensor = torch.tensor([t], device=device)
		
		noised_img, noise = scheduler.q_forward(img_tensor, t_tensor, epsilon=noise)
		pred = model(noised_img, t_tensor)
		
		loss = torch.nn.MSELoss()(pred, noise).detach().item()
		losses[t - 1] = float(loss)
		
		pbar.update(1)
	pbar.close()
	
	x = np.linspace(1, scheduler.T - 1, scheduler.T - 1)
	plt.plot(x, losses, label='Line Std')
	
	plt.xlabel('Diffusion Step')
	plt.ylabel('Loss')
	plt.title('Noise prediction Loss, per Diffusion step')
	
	plt.show()


@torch.no_grad()
def plot_timestep_mean_std(model, scheduler, model_abs_path='final.pth', test_img_path='test.jpg', device='cpu', **kwargs):
	model.load_state_dict(torch.load(model_abs_path, map_location=device))
	model.to(device)
	model.eval()


@hydra.main(config_path="../config", config_name="diffusion", version_base="1.3")
def load_run(cfg: DictConfig):
	model_path, model_name = cfg.model.type.rsplit(".", maxsplit=1)
	model_cls = getattr(importlib.import_module(model_path), model_name)
	model = model_cls(**cfg.model.params)
	
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler_cls = getattr(importlib.import_module(sche_path), sche_name)
	scheduler = scheduler_cls(**cfg.schedule.params)
	
	main(model, scheduler, **cfg.tests.params)
	plot_timestep_loss(model, scheduler, **cfg.tests.params)


if __name__ == '__main__':
	# !!! Remember to set your working dir to the main project dir
	load_run()
