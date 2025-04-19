import torch
import numpy as np


class NoiseScheduler:
	"""
	Base class for noise scheduler. Implementing the methods for a forward noising process.
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

		:param x_0: Batch of image tensors, shape: (b, c, h, w)
		:param t: Batch of time steps, for the diffusion process, shape: (b, 1)
		:param device: Device on which to generate the noise: "cpu"
		:return: Noised image tensors, shape: (b, c, h, w)
		"""
		
		epsilon = torch.rand_like(x_0, device=device)
		x_t = self.sqrt_alpha_bar[t][:, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * epsilon
		
		return x_t, epsilon


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
		# Calculate beta in order to clip it, for numerical stability  # todo: test if necessary
		betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
		betas = torch.clip(betas, 0.0001, 0.9999)
		# Calculate back into alpha_bar from beta
		alphas = 1.0 - betas
		alpha_bars = torch.cumprod(alphas, dim=0)
		
		# Final calculation of sqrt_alphas
		self.sqrt_alpha_bar = alpha_bars ** 0.5
		self.sqrt_one_minus_alpha_bar = (1 - alpha_bars) ** 0.5


class LinearNoiseScheduler(NoiseScheduler):
	"""
	Scheduler for a linear beta schedule. Beta increases linearly from beta_start to beta_end.
	"""
	def __init__(self, n_timesteps=1000, beta_start=0.001, beta_end=0.02):
		super().__init__(n_timesteps)

		# Linearly spaced beta values
		self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
		self.alphas = 1.0 - self.betas
		self.alpha_bars = torch.cumprod(self.alphas, dim=0)

		# Precompute squares for sampling
		self.sqrt_alpha_bar = torch.sqrt(self.alpha_bars)
		self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars)

		# Cache for reverse process
		self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
		self.posterior_variance = (self.betas * (1.0 - torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])) / (1.0 - self.alpha_bars))

	def q_forward(self, x_0, t, device='cpu'):
		"""
        Overrides base method: forward diffusion process that adds Gaussian noise to the input image x_0.
        """
		noise = torch.randn_like(x_0, device=device)
		# Compute square root of alpha bar at given timestep
		sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
		# Compute square root of alpha bar -1 at given timestep
		sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
		return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

	def reverse_diffusion_step(self, x_t, predicted_noise, t):
		"""
        Reverse diffusion process: estimate x_{t-1} from x_t and predicted noise.

        :param x_t: image with noise at time t, shape: [b, c, h, w]
        :param predicted_noise: prediction of noise made by the model, shape: [b, c, h, w]
        :param t: timestep as int (scalar, not a batch)
        :return: tuple of (x_{t-1}, predicted x_0)
        """
		# Ensure scalar timestep
		if isinstance(t, torch.Tensor):
			t = t.item()

		# Estimate x_0 from predicted noise
		sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
		sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]
		x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
		x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

		# Compute posterior mean
		# Noise level at timestep t:
		beta_t = self.betas[t]
		alpha_t = self.alphas[t]
		sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
		# Compute the mean of the reverse distribution
		posterior_mean = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t)

		# No need to add noise at timestep t=0
		if t == 0:
			return posterior_mean, x_0_pred

		# Add stochastic noise to simulate reverse diffusion
		# Sample a random noise vector
		noise = torch.randn_like(x_t)
		variance = self.posterior_variance[t]
		std_dev = torch.sqrt(variance)
		x_prev = posterior_mean + std_dev * noise

		return x_prev, x_0_pred