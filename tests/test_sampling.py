import hydra

from diffusion_lab.models.noise_scheduler import CosineNoiseScheduler, LinearNoiseScheduler, NoiseScheduler
# from diffusion_lab.models.diffusion import UNet
from diffusion_lab.models.unet import UNet
from diffusion_lab.sampling.sampling import sample_image

from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import transforms
from PIL import Image

import torch


@torch.no_grad()
def test_sampling(model, scheduler: NoiseScheduler, n_timesteps=1_000, n_images=1, resolution=(64, 64), start_noise=None):
	model.eval()
	x_t = torch.randn((n_images, 3, *resolution), device=model.device)  # B, C, W, H
	if start_noise is not None:
		x_t = start_noise
	
	for t in reversed(range(1, n_timesteps)):
		t_tensor = torch.full((n_images,), t, device=model.device, dtype=torch.long)
		epsilon = model(x_t, t_tensor)
		x_t, _ = scheduler.p_backward(x_t, epsilon, t_tensor)
	return x_t


@hydra.main(config_path="../../config", config_name="diffusion", version_base="1.3")
def main():
	# DO NOT CHANGE! Read comment on the bottom!
	cfg = OmegaConf.load("config/model/unet.yaml")
	# DO NOT CHANGE! Read comment on the bottom! 9
	path = 'data/resized_images/Cat/cat-test_(1).jpeg'
	img = Image.open(path).convert('RGB')
	
	device = 'mps'
	# model = UNet(cfg=cfg, device=device)
	model = UNet(device=device)
	# model.load_state_dict(torch.load("/Volumes/Filip'sTech/outputs/2025-05-09/00-36-26/steps/step-350.pth", map_location=device))
	model.load_state_dict(torch.load("/Volumes/Filip'sTech/step-110.pth", map_location=device))
	model.to(device)
	model.eval()
	
	img = to_tensor(img)
	img = img.to(device)
	img.unsqueeze(0)
	
	# for name, param in model.state_dict().items():
	# 	print(name, param.shape)
	
	with torch.no_grad():
		scheduler = CosineNoiseScheduler(n_timesteps=1_000, device=device)
		noise = torch.randn_like(img, device=device)
		x_t, _ = scheduler.q_forward(img, torch.tensor([999]), epsilon=noise)
		
		epsilon_t = model(x_t, torch.tensor([999], device=model.device))
		x_t_m_1, x_0 = scheduler.p_backward(x_t, epsilon_t, torch.tensor([999]))
		
		bgc = Image.new('RGB', (64 * 3, 64), color='white')
		
		bgc.paste(to_pil(x_t[0]), (0, 0))
		bgc.paste(to_pil(x_0[0]), (64, 0))
		
		sampled = sample_image(model, scheduler, n_timesteps=1_000, n_images=1, resolution=(64, 64))
		bgc.paste(to_pil(sampled[0]), (128, 0))
	bgc.show()


if __name__ == '__main__':
	main()
