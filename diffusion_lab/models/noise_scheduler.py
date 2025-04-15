import torch
import numpy as np


class NoiseScheduler:
	"""
	Base class for noise scheduler. Implementing the methods for forward noising process.
	"""
	def __init__(self, n_timesteps=1000):
		self.T = n_timesteps
		
		# to be overwritten by children
		self.sqrt_alpha_bar = torch.zeros(n_timesteps)
		self.sqrt_one_minus_alpha_bar = torch.zeros(n_timesteps)
	
	def q_forward(self, x_0, t, device='cpu'):
		"""
		Forward diffusion process, adding Gaussian noise to the image :math:`x_0`.

		Forward: :math:`q(x_t | x_0 ) = \\mathcal{N}(x_t;\\sqrt{\\alpha_t}x_0, \\sqrt{(1-\\alpha_t)}\\textbf{I})`

		Noising: :math:`x_t = \\sqrt{\\alpha_t}x_0 + \sqrt{1-\\alpha_t}\\epsilon`

		:param x_0: batch of image tensors, shape: (b, c, h, w)
		:param t: batch of time steps, for the diffusion process, shape: (b, 1)
		:param device: device on which to generate the noise: "cpu"
		:return: noised image tensors, shape: (b, c, h, w)
		"""
		
		epsilon = torch.rand_like(x_0, device=device)
		x_t = self.sqrt_alpha_bar[t][:, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * epsilon
		
		return x_t


class CosineNoiseScheduler(NoiseScheduler):
	"""
	Scheduler for cosine schedule proposed in "Improved Denoising Diffusion Probabilistic Models".
	Cosine schedule adds less noise at the initial and final timestamps, while staying linear in the middle.
	"""
	def __init__(self, n_timesteps=1000, s=0.008):
		super().__init__(n_timesteps)
		
		x = torch.linspace(0, n_timesteps, n_timesteps + 1)
		# Calculate alpha_bar according to the formula
		alphas_bar = torch.cos(((x / n_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
		alphas_bar = alphas_bar / alphas_bar[0]
		# Calculate beta in order to clip it, for numerical stability
		betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
		betas = torch.clip(betas, 0.0001, 0.9999)
		# Calculate back into alpha_bar from beta
		alphas = 1.0 - betas
		alpha_bars = torch.cumprod(alphas, dim=0)
		# Final calculation of sqrt_alphas
		self.sqrt_alpha_bar = alpha_bars ** 0.5
		self.sqrt_one_minus_alpha_bar = (1 - alpha_bars) ** 0.5
