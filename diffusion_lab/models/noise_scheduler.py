import torch
import numpy as np


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
	
	def q_forward(self, x_0, t, epsilon=None):
		"""
		Forward diffusion process, adding Gaussian noise to the image :math:`x_0`.

		Forward: :math:`q(x_t | x_0 ) = \\mathcal{N}(x_t;\\sqrt{\\alpha_t}x_0, \\sqrt{(1-\\alpha_t)}\\textbf{I})`

		Noising: :math:`x_t = \\sqrt{\\alpha_t}x_0 + \sqrt{1-\\alpha_t}\\epsilon`

		:param x_0: Batch of image tensors, shape: (b, c, h, w)
		:param t: Batch of time steps, for the diffusion process, shape: (b, 1)
		:return: Noised image tensors, shape: (b, c, h, w)
		"""
		
		if epsilon is None:
			epsilon = torch.randn_like(x_0, device=self.device)
		x_t = (torch.sqrt(self.alpha_bars[t][:, None, None, None]) * x_0 +
		       torch.sqrt(1 - self.alpha_bars[t][:, None, None, None]) * epsilon)
		
		return x_t, epsilon
	
	def p_backward(self, x_t, epsilon_t, t):
		batch_size = x_t.size(0)
		t = t.view(-1) if t.ndim == 0 else t
		
		z = torch.randn_like(x_t, device=self.device) if t > 1 else torch.zeros_like(x_t, device=self.device)
		
		beta_t = self.betas[t].view(batch_size, 1, 1, 1)
		alpha_t = self.alphas[t].view(batch_size, 1, 1, 1)
		alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1, 1)
		alpha_bar_tm1 = self.alpha_bars[t - 1].view(batch_size, 1, 1, 1)
		
		x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * epsilon_t) / torch.sqrt(alpha_bar)
		img = x_0_pred[0][0].cpu().detach().numpy()
		# x_0_pred = torch.clamp(x_0_pred, min=-1, max=1)
		
		x_t_prev = torch.sqrt(alpha_bar_tm1) * x_0_pred + torch.sqrt(1 - alpha_bar_tm1) * z
		# x_t_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / (torch.sqrt(alpha_bar)) * epsilon_t) + torch.sqrt(1-alpha_t) * z
		# x_t_prev = torch.clamp(x_t_prev, -1, 1)
		# pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
		img = x_t_prev[0][0].cpu().detach().numpy()
		
		return x_t_prev, x_0_pred


class CosineNoiseScheduler(NoiseScheduler):
	"""
	Scheduler for cosine schedule proposed in "Improved Denoising Diffusion Probabilistic Models".
	Cosine schedule adds less noise at the initial and final timestamps, while staying linear in the middle.
	"""
	
	def __init__(self, n_timesteps=1000, s=0.008, device="cpu"):
		super().__init__(n_timesteps, device)
		x = torch.arange(0, n_timesteps + 1, device=device)
		
		# Calculate alpha_bar according to the formula
		alphas_bar = torch.cos(((x / n_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
		alphas_bar = alphas_bar / alphas_bar[0]
		# Calculate beta in order to clip it, for numerical stability  # todo: test if necessary
		betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
		betas = torch.clip(betas, 0.0001, 0.9999)
		# Calculate back into alpha_bar from beta
		alphas = 1.0 - betas
		# alphas_bar = torch.cumprod(alphas, dim=0)
		
		self.betas = betas
		self.alphas = alphas
		self.alpha_bars = alphas_bar[:-1]


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
	
	path = 'data/resized_images/Cat/cat-test_(1).jpeg'
	image = Image.open(path).convert('RGB')
	
	train_transformation = transforms.Compose([
		transforms.Resize(64),
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
	
	bgc = Image.new('RGB', (64 * (11 + 1), 64 * 3), color='white')
	
	scheduler = CosineNoiseScheduler(n_timesteps=1000)
	
	x_0 = train_transformation(image)
	x_0 = x_0.unsqueeze(0)
	
	bgc.paste(to_pil(x_0[0]), (0, 0))
	
	for t in reversed(list(range(0, 1000, 100)) + [999]):
		# t = 600
		
		x_t, epsilon = scheduler.q_forward(x_0, torch.tensor([t]))
		bgc.paste(to_pil(x_t[0]), (64 + 64 * ((t + 1) // 100), 0))
		
		x_prev, x_0_pred = scheduler.p_backward(x_t, epsilon, torch.tensor([t]))
		bgc.paste(to_pil(x_0_pred[0]), (64 + 64 * ((t + 1) // 100), 64))
		bgc.paste(to_pil(x_prev[0]), (64 + 64 * ((t + 1) // 100), 128))
	
	bgc.show()
