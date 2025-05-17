from diffusion_lab.models.noise_scheduler import CosineNoiseScheduler, LinearNoiseScheduler
from diffusion_lab.models.diffusion import UNet
from diffusion_lab.sampling.sampling import sample_image

from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import transforms
from PIL import Image

import torch


def main():
	# DO NOT CHANGE! Read comment on the bottom!
	cfg = OmegaConf.load("config/model/unet.yaml")
	# DO NOT CHANGE! Read comment on the bottom! 9
	path = 'data/resized_images/Cat/cat-test_(1).jpeg'
	img = Image.open(path).convert('RGB')
	
	to_tensor = transforms.Compose([
		transforms.Resize(64),
		transforms.ToTensor(),
		transforms.Normalize(  # Normalize RGB pixel values: [0, 255] -> [-1, 1]
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5]
		)
	])
	to_pil = transforms.Compose([
		transforms.Normalize(  # Normalize RGB pixel values: [-1, 1] -> [0, 1]
			mean=[-1.0, -1.0, -1.0],
			std=[2.0, 2.0, 2.0]
		),
		transforms.ToPILImage(),
	])
	
	device = 'mps'
	model = UNet(cfg=cfg, device=device)
	# model.load_state_dict(torch.load("/Volumes/Filip'sTech/outputs/2025-05-09/00-36-26/steps/step-350.pth", map_location=device))
	model.load_state_dict(torch.load("/Volumes/Filip'sTech/outputs/2025-05-08/12-04-20/steps/step-300.pth", map_location=device))
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
		x_t, _ = scheduler.q_forward(img, torch.tensor([900]), epsilon=noise)
		
		epsilon_t = model(x_t, torch.tensor([900], device=model.device))
		x_t_m_1, x_0 = scheduler.p_backward(x_t, epsilon_t, torch.tensor([900]))
		
		bgc = Image.new('RGB', (64 * 3, 64), color='white')
		
		bgc.paste(to_pil(x_t[0]), (0, 0))
		bgc.paste(to_pil(x_0[0]), (64, 0))
		
		sampled = sample_image(model, scheduler, n_timesteps=1_000, n_images=1, resolution=(64, 64))
		bgc.paste(to_pil(sampled[0]), (128, 0))
	bgc.show()


if __name__ == '__main__':
	main()
