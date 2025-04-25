import torch
import numpy as np


class NoiseScheduler:
	"""
	Base class for noise scheduler. Implementing the methods for a forward noising process.
	"""
	def __init__(self, n_timesteps=1000, device="cpu"):
		"""
		todo: description
		:param n_timesteps: 
		:param device: Device on which to generate the noise: "cpu"
		"""
		self.T = n_timesteps
		self.device = device
		
		# to be overwritten by children
		self.sigma = torch.zeros(n_timesteps, device=self.device)
		self.one_over_sqrt_alpha = torch.zeros(n_timesteps, device=self.device)
		self.sqrt_alpha_bar = torch.zeros(n_timesteps, device=self.device)
		self.sqrt_one_minus_alpha_bar = torch.zeros(n_timesteps, device=self.device)
		self.one_m_alpha_over_sqrt_one_m_alpha_bar = torch.zeros(n_timesteps, device=self.device)
	
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
		x_t = self.sqrt_alpha_bar[t][:, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * epsilon
		
		return x_t, epsilon
	
	def p_backward(self, x_t, epsilon_t, t):
		z = torch.randn_like(x_t, device=self.device) if t > 1 else torch.zeros_like(x_t, device=self.device)
		x_t_minus_noise = x_t - self.one_m_alpha_over_sqrt_one_m_alpha_bar[t][:, None, None, None] * epsilon_t
		x_t_minus_one = self.one_over_sqrt_alpha[t][:, None, None, None] * x_t_minus_noise + self.sigma[t][:, None, None, None] * z
		
		return x_t_minus_one


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
		
		# Final calculation to save
		self.sigma = betas ** 0.5
		self.one_over_sqrt_alpha = 1 / alphas ** 0.5
		self.sqrt_alpha_bar = alpha_bars ** 0.5
		self.sqrt_one_minus_alpha_bar = (1 - alpha_bars) ** 0.5
		self.one_m_alpha_over_sqrt_one_m_alpha_bar = (1 - alphas) / self.sqrt_one_minus_alpha_bar


class LinearNoiseScheduler(NoiseScheduler):
	"""
	Scheduler for a linear beta schedule. Beta increases linearly from beta_start to beta_end.
	"""
	def __init__(self, n_timesteps=1000, beta_start=0.001, beta_end=0.02, device="cpu"):
		super().__init__(n_timesteps, device)

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

	def q_forward(self, x_0, t):
		"""
        Overrides base method: forward diffusion process that adds Gaussian noise to the input image x_0.
        """
		noise = torch.randn_like(x_0, device=self.device)
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
