import importlib

import hydra
import torch
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from diffusion_lab.models.noise_scheduler import NoiseScheduler
from diffusion_lab.utils.transforms import to_pil, to_tensor


def test_schedule(scheduler: NoiseScheduler, image, t):
	x_0 = to_tensor(image)
	x_0 = x_0.to(scheduler.device)
	x_0 = x_0.unsqueeze(0)
	
	t = torch.tensor([t], device=scheduler.device)
	
	torch.manual_seed(0)
	noise = torch.randn(x_0.shape, device=scheduler.device)
	
	x_t, _ = scheduler.q_forward(x_0, t, epsilon=noise)
	x_prev, x_0_pred = scheduler.p_backward(x_t, noise, t)
	
	return to_pil(x_t[0]), to_pil(x_prev[0]), to_pil(x_0_pred[0])


@torch.no_grad()
def plot_step_mean_std(scheduler: NoiseScheduler, image):
	img_tensor = to_tensor(image)
	img_tensor = img_tensor.to(scheduler.device)
	img_tensor = img_tensor.unsqueeze(0)
	img_tensor = img_tensor.repeat(scheduler.T, 1, 1, 1)
	
	t = torch.arange(scheduler.T, device=scheduler.device)
	
	x_t, _ = scheduler.q_forward(img_tensor, t)
	mean = x_t.mean(dim=(1, 2, 3)).cpu().numpy()
	std = x_t.std(dim=(1, 2, 3)).cpu().numpy()
	
	x = t.cpu().numpy()
	plt.plot(x, mean, label='Line Mean')
	plt.plot(x, std, label='Line Std')
	
	plt.xlabel('Diffusion Step')
	plt.ylabel('Y-axis')
	plt.title('Mean and Std by Diffusion Step')
	
	plt.legend()
	plt.show()


def main(scheduler, test_img_path='test.jpg', show_steps=None, **kwargs):
	if show_steps is None:
		show_steps = [1, 500, 999]
	
	image = Image.open(test_img_path).convert('RGB')
	
	n_cols = len(show_steps)
	
	bgc = Image.new('RGB', (64 * n_cols, 64 * 2), color='white')
	for i, t in enumerate(show_steps):
		noised_img, _, x_0_approx = test_schedule(scheduler, image, t)
		
		bgc.paste(noised_img, (64 * i, 0))
		bgc.paste(x_0_approx, (64 * i, 64))
	
	plot_step_mean_std(scheduler, image)
	
	bgc.show()


@hydra.main(config_path="../config", config_name="diffusion", version_base="1.3")
def load_run(cfg: DictConfig):
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler_cls = getattr(importlib.import_module(sche_path), sche_name)
	scheduler = scheduler_cls(**cfg.schedule.params)
	
	main(scheduler, **cfg.tests.params)
	

if __name__ == '__main__':
	# !!! Remember to set your working dir to the main project dir
	load_run()


