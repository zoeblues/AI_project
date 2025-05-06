import torch
from PIL import Image
from torchvision import transforms
from diffusion_lab.models.noise_scheduler import CosineNoiseScheduler, NoiseScheduler, LinearNoiseScheduler
from diffusion_lab.models.diffusion import UNet
from omegaconf import OmegaConf


def sample_from_scheduler(model, scheduler: NoiseScheduler, n_timesteps=1000, resolution=(128, 128), device="cpu"):
	model.eval()
	with torch.no_grad():
		x_t = torch.randn((1, 3, *resolution), device=model.device)
		for t in reversed(range(1, n_timesteps)):
			t_tensor = torch.full((1,), t, dtype=torch.long, device=model.device)
			epsilon_pred = model(x_t, t_tensor)
			x_t, _ = scheduler.p_backward(x_t, epsilon_pred, t_tensor)
	
	to_pil = transforms.Compose([
		transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
		transforms.ToPILImage()
	])
	x_img = x_t[0].cpu().clamp(-1, 1)  # Extract single image (C, H, W)
	return to_pil(x_img)



if __name__ == "__main__":
	device = 'cpu'
	resolution = (128, 128)

	to_pil = transforms.Compose([
		transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
		transforms.ToPILImage(),
	])

	# Load model config and model
	cfg = OmegaConf.load("../../config/model/unet.yaml")
	model = UNet(cfg=cfg, device=device)
	model.load_state_dict(torch.load("outputs/uncond_unet.pth", map_location=device))
	model.to(device)
	model.eval()

	scheduler = CosineNoiseScheduler(n_timesteps=1000, device=device)

	# Start from noise
	x_t = torch.randn((1, 3, 128, 128), device=device)
	timestep = torch.tensor([999], device=device)
	epsilon_t = model(x_t, timestep)

	# One reverse step
	x_t_m_1, x_0_pred = scheduler.p_backward(x_t, epsilon_t, timestep)

	# Full sampling
	sampled_img = sample_from_scheduler(model, scheduler, n_timesteps=900, resolution=resolution, device=device)
	
	canvas_width = 128 * 3
	canvas_height = 128
	bgc = Image.new('RGB', (canvas_width, canvas_height), color='white')
	
	# Convert tensors to PIL images
	initial_noise_img = to_pil(x_t[0].cpu().clamp(-1, 1))
	predicted_x0_img = to_pil(x_0_pred[0].cpu().clamp(-1, 1))
	final_sampled_img = sampled_img  # already PIL
	
	# Paste them side by side
	bgc.paste(initial_noise_img, (0, 0))  # First slot
	bgc.paste(predicted_x0_img, (128, 0))  # Second slot
	bgc.paste(final_sampled_img, (256, 0))
	
	bgc.show()