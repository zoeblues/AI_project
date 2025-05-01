import math
import numpy as np
import torch
import torch.nn as nn

from diffusion_lab.models.noise_scheduler import NoiseScheduler


def sample_image(model, scheduler: NoiseScheduler, n_timesteps=1_000, n_images=1, resolution=(64, 64)):
	model.eval()
	x_t = torch.randn((n_images, 3, *resolution), device=model.device)  # B, C, W, H
	with torch.no_grad():
		for t in reversed(range(n_timesteps)):
			t = torch.tensor([t], device=model.device)
			epsilon = model(x_t, t)
			x_t = scheduler.p_backward(x_t, epsilon, t)
	return x_t


if __name__ == '__main__':
	from diffusion_lab.models.noise_scheduler import CosineNoiseScheduler
	from diffusion_lab.models.diffusion import UNet
	from omegaconf import DictConfig, OmegaConf
	
	from torchvision.transforms import transforms
	from PIL import Image
	
	# todo: use unet config
	'''
		cfg = DictConfig({
		'base_channels': 64,
		'image_size': 128,
		'in_channels': 3,
		'num_groups': 32,
		'time_embedding_factor': 4,
		'attention_heads': 4,
		'out_channels': 3,
		'num_layers': 2
	})
	
	'''

	to_pil = transforms.ToPILImage()
	
	cfg = OmegaConf.load("../../config/model/unet.yaml")
	
	# Initialize the model
	device = 'cpu'
	model = UNet(cfg=cfg, device=device)
	model.load_state_dict(torch.load("uncond_unet.pth", map_location='cpu'))
	model.to(device)
	scheduler = CosineNoiseScheduler(1000, device=model.device)
	
	x_t = torch.randn((1, 3, 64, 64), device=model.device)  # B, C, W, H
	epsilon_t = model(x_t, torch.tensor([999], device=model.device))
	x_0 = x_t - epsilon_t
	
	image_tensor = x_0[0]
	
	bgc = Image.new('RGB', (64 * 3, 64), color='white')
	
	bgc.paste(to_pil(x_t[0]), (0, 0))
	bgc.paste(to_pil(epsilon_t[0]), (64, 0))
	bgc.paste(to_pil(image_tensor), (128, 0))
	bgc.show()
