import math

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from diffusion_lab.models.diffusion import UNet
from diffusion_lab.models.noise_scheduler import NoiseScheduler, CosineNoiseScheduler, LinearNoiseScheduler


@torch.no_grad()
def predict(model, scheduler, img, timestep):
	pred = model(img, timestep)
	less_noised_img, clear_img = scheduler.p_backward(img, pred, timestep)


@torch.no_grad()
def test_denoise(model: UNet, scheduler: NoiseScheduler, img_tensor: torch.Tensor, timestep: int, device='cpu'):
	img_tensor = img_tensor.to(device)
	if not isinstance(timestep, torch.Tensor):
		timestep = torch.tensor([timestep], device=device)
	
	# Change to batch form (shape) -> (1, shape)
	img_tensor = img_tensor.unsqueeze(0)
	
	noise = torch.randn_like(img_tensor, device=device)
	
	noised_img, noise = scheduler.q_forward(img_tensor, timestep, epsilon=noise)
	# img = noise[0][0].cpu().numpy()
	# img = noise[0][1].cpu().numpy()
	# img = noise[0][2].cpu().numpy()
	pred = model(noised_img, timestep)
	less_noised_img, clear_img = scheduler.p_backward(noised_img, pred, timestep)
	
	return noised_img[0], noise[0], pred[0], less_noised_img[0], clear_img[0]


def main():
	# DO NOT CHANGE! Read comment on the bottom!
	cfg = OmegaConf.load("config/model/unet.yaml")
	# DO NOT CHANGE! Read comment on the bottom! 9
	path = 'data/resized_images/Cat/cat-test_(1).jpeg'
	image = Image.open(path).convert('RGB')
	
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
	
	# for name, param in model.state_dict().items():
	# 	print(name, param.shape)
	
	scheduler = CosineNoiseScheduler(n_timesteps=1_000, device=device)
	
	step = 100
	
	n_rows = 3
	n_columns = 1000//step + 1
	
	bgc = Image.new("RGB", (64 * n_columns, 64 * n_rows), color=(255, 255, 255)).convert("RGB")
	bgc.paste(image.resize((64, 64)), (0, 0))
	
	for t in reversed(list(range(1, 1000, step)) + [999]):
		noised_img, noise, pred_noise, _, pred_img = test_denoise(model, scheduler, to_tensor(image), t, device=device)
		img = pred_img[0].cpu().numpy()
		img = pred_img[1].cpu().numpy()
		img = pred_img[2].cpu().numpy()
		loss = torch.nn.MSELoss()(pred_img, noise)
		
		print(t, loss.item())
		i = (t + 1) // step
		
		bgc.paste(to_pil(noised_img), (64 * (i + 1), 0))
		bgc.paste(to_pil(noise - pred_noise), (64 * (i + 1), 64))
		bgc.paste(to_pil(pred_img), (64 * (i + 1), 128))
	
	bgc.show()


if __name__ == '__main__':
	# !!! Remember to set your working dir to the main project dir
	main()
