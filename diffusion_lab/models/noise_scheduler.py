import torch
import numpy as np
from torchvision import transforms


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
	
	def p_backward(self, x_t, epsilon_theta, t, epsilon_t=None):
		batch_size = x_t.size(0)
		t = t.view(-1) if t.ndim == 0 else t
		
		if epsilon_t is None:
			z = torch.randn_like(x_t, device=self.device)
		else:
			z = epsilon_t
		z[t == 0] = 0
		
		beta_t = self.betas[t].view(batch_size, 1, 1, 1)
		alpha_t = self.alphas[t].view(batch_size, 1, 1, 1)
		alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1, 1)
		alpha_bar_tm1 = self.alpha_bars[t - 1].view(batch_size, 1, 1, 1)
		
		x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * epsilon_theta) / torch.sqrt(alpha_bar)
		# x_0_pred = torch.clamp(x_0_pred, -1, 1)
		
		# test = x_0_pred[0][0].cpu().detach().numpy()
		# x_0_pred = torch.clamp(x_0_pred, min=-1, max=1)
		
		# normalization = transforms.Normalize(mean=x_0_pred.mean(), std=x_0_pred.std())
		# x_0_pred = normalization(x_0_pred)
		# test = x_0_pred[0][0].cpu().detach().numpy()
		
		x_t_prev = torch.sqrt(alpha_bar_tm1) * x_0_pred + torch.sqrt(1 - alpha_bar_tm1) * z
		# x_t_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / (torch.sqrt(alpha_bar)) * epsilon_t) + torch.sqrt(1-alpha_t) * z
		# x_t_prev = torch.clamp(x_t_prev, -1, 1)
		# pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
		# img = x_t_prev[0][0].cpu().detach().numpy()
		
		# normalization = transforms.Normalize(mean=x_t_prev.mean(), std=x_t_prev.std())
		# x_t_prev = normalization(x_t_prev)
		# img = x_t_prev[0][0].cpu().detach().numpy()
		
		# x_tm1 = 1 / torch.sqrt(alpha_t) * (x_t - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar))) * epsilon_t) + torch.sqrt(beta_t) * epsilon_t
		
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
