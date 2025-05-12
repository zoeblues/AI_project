import torch
import numpy as np
from diffusion import UNet

class NoiseScheduler:
	"""
	Base class for noise scheduler. Implementing the methods for a forward noising process.
	"""
	
	def __init__(self, n_timesteps=1000, device="cpu"):
		"""
		:param n_timesteps: 
		:param device: Device on which to generate the noise: "cpu"
		"""
		self.T = n_timesteps
		self.device = device
		
		self.betas = torch.zeros(n_timesteps, device=self.device)
		self.alphas = torch.zeros(n_timesteps, device=self.device)
		self.alpha_bars = torch.zeros(n_timesteps, device=self.device)
		
	def q_forward(self, x_0, t):
		"""
		Forward diffusion process, adding Gaussian noise to the image :math:`x_0`.

		Forward: :math:`q(x_t | x_0 ) = \\mathcal{N}(x_t;\\sqrt{\\alpha_t}x_0, \\sqrt{(1-\\alpha_t)}\\textbf{I})`

		Noising: :math:`x_t = \\sqrt{\\alpha_t}x_0 + \sqrt{1-\\alpha_t}\\epsilon`

		:param x_0: Batch of image tensors, shape: (b, c, h, w)
		:param t: Batch of time steps, for the diffusion process, shape: (b, 1)
		:return: Noised image tensors, shape: (b, c, h, w)
		"""
		
		epsilon = torch.randn_like(x_0, device=self.device)
		x_t = (torch.sqrt(self.alpha_bars[t][:, None, None, None]) * x_0 +
		       torch.sqrt(1 - self.alpha_bars[t][:, None, None, None]) * epsilon)
		
		return x_t, epsilon
	
	def p_backward(self, x_t, epsilon_t, t):
		
		if t.ndim == 0:
			t = t.expand(x_t.size(0))
		elif t.ndim == 1 and t.size(0) == 1:
			t = t.expand(x_t.size(0))
		
		z = torch.randn_like(x_t, device=self.device)
		z = torch.where((t > 0)[:, None, None, None], z, torch.zeros_like(z))
	
		beta_t = self.betas[t][:, None, None, None]
		alpha_t = self.alphas[t][:, None, None, None]
		alpha_bar_t = self.alpha_bars[t][:, None, None, None]
		
		#x_t_minus_one = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar) * epsilon_t) / torch.sqrt(alpha_t)
		# x_t_minus_one = torch.clamp(x_t_minus_one, -1.0, 1.0)
		
		#pred_x_0 = (x_t - (1 - alpha_bar) / torch.sqrt(1 - alpha_bar) * epsilon_t) / torch.sqrt(alpha_t)
		# pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
		x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_t) / torch.sqrt(alpha_bar_t)
		
		# Step 2: Estimate mean of p(x_{t-1} | x_t)
		mean = (1 / torch.sqrt(alpha_t)) * (
				x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_t
		)
		
		# Step 3: Add noise
		x_t_minus_1 = mean + torch.sqrt(beta_t) * z if t[0] > 0 else mean
		
		#return x_t_minus_one + torch.sqrt(beta_t) * z, pred_x_0
		return x_t_minus_1, x_0_pred


class CosineNoiseScheduler(NoiseScheduler):
	"""
	Scheduler for cosine schedule proposed in "Improved Denoising Diffusion Probabilistic Models".
	Cosine schedule adds less noise at the initial and final timestamps, while staying linear in the middle.
	"""
	
	def __init__(self, n_timesteps=1000, s=0.008, device="cpu"):
		super().__init__(n_timesteps, device)
		x = torch.linspace(0, n_timesteps, n_timesteps + 1, device=device)
		
		# Calculate alpha_bar according to the formula
		alphas_bar = torch.cos(((x / n_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
		alphas_bar = alphas_bar / alphas_bar[0]
		# Calculate beta in order to clip it, for numerical stability  # todo: test if necessary
		betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
		betas = torch.clip(betas, 0.0001, 0.9999)
		# Calculate back into alpha_bar from beta
		alphas = 1.0 - betas
		alpha_bars = torch.cumprod(alphas, dim=0)
		
		self.betas = betas
		self.alphas = alphas
		self.alpha_bars = alpha_bars
		

class LinearNoiseScheduler(NoiseScheduler):
	"""
	Scheduler for a linear beta schedule. Beta increases linearly from beta_start to beta_end.
	"""
	
	def __init__(self, n_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
		super().__init__(n_timesteps, device)
		
		# Linearly spaced beta values
		self.betas = torch.linspace(beta_start, beta_end, n_timesteps, device=self.device)
		self.alphas = 1.0 - self.betas
		self.alpha_bars = torch.cumprod(self.alphas, dim=0)
		

if __name__ == "__main__":
	from PIL import Image
	from torchvision import transforms
	from omegaconf import DictConfig, OmegaConf
	
	path = '../data/resized_images/Cat/cat-test_(1).jpeg'
	image = Image.open(path).convert('RGB')
	cfg = OmegaConf.load("../../config/model/unet.yaml")
	model = UNet(cfg, device='cpu')
	model.load_state_dict(torch.load('../sampling/outputs/uncond_unet.pth'))
	model.eval()
	
	train_transformation = transforms.Compose([
		transforms.Resize(128),
		# transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.2)),
		# transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),  # Convert image to tensor
		transforms.Normalize(  # Normalize RGB pixel values: [0, 1] -> [-1, 1]
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
	
	bgc = Image.new('RGB', (128 * 11, 128 * 2), color='white')
	
	scheduler = CosineNoiseScheduler()
	
	x_0 = train_transformation(image)
	x_0 = x_0.unsqueeze(0)
	
	bgc.paste(to_pil(x_0[0]), (0, 0))
	
	for t in range(0, 1000, 100):
		# t = 600
		t_batch = torch.full((1,), t, dtype=torch.long, device=scheduler.device)
		
		x_t, epsilon = scheduler.q_forward(x_0, t_batch)
		bgc.paste(to_pil(x_t[0].clamp(-1, 1)), (128 + 128 * (t // 100), 0))
		
		predicted_epsilon = model(x_t, torch.tensor([t], device='cpu'))
		_, x_0_pred = scheduler.p_backward(x_t, predicted_epsilon, torch.tensor([t]))
		bgc.paste(to_pil(x_0_pred[0].clamp(-1, 1)), (128 + 128 * (t // 100), 128))
	bgc.show()
	
