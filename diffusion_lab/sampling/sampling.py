import math
import numpy as np
import torch
import torch.nn as nn

from diffusion_lab.models.noise_scheduler import NoiseScheduler


def sample_image(model, scheduler: NoiseScheduler, n_timesteps=1_000, n_images=1, resolution=(128, 128)):
	model.eval()
	x_t = torch.randn((n_images, 3, *resolution), device=model.device)  # B, C, W, H
	with torch.no_grad():
		for t in reversed(range(1, n_timesteps)):
			print(t)
			t = torch.tensor([t], device=model.device)
			epsilon = model(x_t, t)
			x_t, _ = scheduler.p_backward(x_t, epsilon, t)
	return x_t


if __name__ == '__main__':
	from diffusion_lab.models.noise_scheduler import CosineNoiseScheduler, LinearNoiseScheduler
	from diffusion_lab.models.diffusion import UNet
	from omegaconf import DictConfig, OmegaConf
	
	from torchvision.transforms import transforms
	from PIL import Image
	
	# todo: use unet config
	'''
		cfg = DictConfig({
		'base_channels': 128,
		'image_size': 128,
		'in_channels': 3,
		'num_groups': 32,
		'time_embedding_factor': 4,
		'attention_heads': 4,
		'out_channels': 3,
		'num_layers': 2
	})
	
	'''
	
	to_pil = transforms.Compose([
		transforms.Normalize(  # Normalize RGB pixel values: [-1, 1] -> [0, 1]
			mean=[-1.0, -1.0, -1.0],
			std=[2.0, 2.0, 2.0]
		),
		transforms.ToPILImage(),
	])
	
	cfg = OmegaConf.load("../../config/model/unet.yaml")
	
	# Initialize the model
	device = 'cpu'
	model = UNet(cfg=cfg, device=device)
	model.load_state_dict(torch.load("outputs/uncond_unet.pth", map_location=device))
	model.to(device)
	scheduler = CosineNoiseScheduler(1000, device=model.device)
	
	x_t = torch.randn((1, 3, 128, 128), device=model.device)  # B, C, W, H
	epsilon_t = model(x_t, torch.tensor([999], device=model.device))
	x_t_m_1, x_0 = scheduler.p_backward(x_t, epsilon_t, torch.tensor([999]))
	sampled = sample_image(model, scheduler, n_timesteps=1_000, n_images=1, resolution=(128, 128))
	
	bgc = Image.new('RGB', (128 * 4, 128), color='white')
	
	bgc.paste(to_pil(x_t[0]), (0, 0))
	bgc.paste(to_pil(epsilon_t[0]), (128, 0))
	bgc.paste(to_pil(x_0[0]), (256, 0))
	bgc.paste(to_pil(sampled[0]), (384, 0))
	bgc.show()
