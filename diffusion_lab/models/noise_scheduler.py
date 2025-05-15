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
	
	def p_backward(self, x_t, epsilon_t, t):
		batch_size = x_t.size(0)
		t = t.view(-1) if t.ndimension() == 0 else t
		
		# Ensure z is sampled only if t > 0, otherwise use zeros at t == 0
		z = torch.randn_like(x_t, device=self.device) if t > 0 else torch.zeros_like(x_t, device=self.device)
		
		# Ensure proper indexing for betas, alphas, and alpha_bars
		beta_t = self.betas[t].view(batch_size, 1, 1, 1)
		alpha_t = self.alphas[t].view(batch_size, 1, 1, 1)
		alpha_bar_t = self.alpha_bars[t].view(batch_size, 1, 1, 1)
		
		# Make sure alpha_bar_tm1 is properly indexed, adjusting for t-1
		alpha_bar_tm1 = self.alpha_bars[t - 1].view(batch_size, 1, 1, 1) if t > 0 else torch.ones_like(alpha_bar_t)
		
		# Predicted x_0 (the clean image estimate)
		x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_t) / torch.sqrt(alpha_bar_t)
		
		# Compute the previous timestep image (x_t_prev)
		x_t_prev = torch.sqrt(alpha_bar_tm1) * x_0_pred + torch.sqrt(1 - alpha_bar_tm1) * z
		
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
	import torch
	
	
	def check_noise_prediction(scheduler, x_0, t_check=100, epsilon_manual=None, verbose=True):
		"""
		Check if the predicted noise from the backward process matches the manually added noise.

		:param scheduler: The noise scheduler (e.g., CosineNoiseScheduler)
		:param x_0: Original clean image tensor (shape: [1, 3, 128, 128])
		:param t_check: Timestep at which to perform the check
		:param epsilon_manual: Optional tensor of the same shape as x_0 to use as manual noise
		:param verbose: Whether to print the results
		:return: Boolean indicating if the noise is recovered accurately, and the error
		"""
		device = scheduler.device
		t_tensor = torch.tensor([t_check], dtype=torch.long, device=device)
		
		if epsilon_manual is None:
			epsilon_manual = torch.randn_like(x_0)
		
		# Forward: add known noise
		x_t, _ = scheduler.q_forward(x_0, t_tensor, epsilon=epsilon_manual)
		
		# Backward: get predicted x_0 and reconstruct predicted epsilon
		_, x_0_pred = scheduler.p_backward(x_t, epsilon_manual, t_tensor)
		
		# Predict epsilon from the reconstructed x_0
		alpha_bar_t = scheduler.alpha_bars[t_check]
		sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
		sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
		
		# Reconstruct predicted noise: epsilon_pred = (x_t - sqrt(alpha_bar) * x_0_pred) / sqrt(1 - alpha_bar)
		epsilon_pred = (x_t - sqrt_alpha_bar * x_0_pred) / sqrt_one_minus_alpha_bar
		
		# Compare
		error = torch.nn.functional.mse_loss(epsilon_pred, epsilon_manual).item()
		match = error < 1e-5  # Adjust tolerance as needed
		
		if verbose:
			print(f"[t={t_check}] MSE between manual and predicted epsilon: {error:.6f}")
			print("✔ Match" if match else "✘ Mismatch")
		
		return match, error
	
	
	# Load and preprocess the image
	path = '../data/resized_images/Cat/cat-test_(1).jpeg'
	image = Image.open(path).convert('RGB')
	
	train_transformation = transforms.Compose([
		transforms.Resize(128),
		transforms.ToTensor(),  # Convert image to tensor
		transforms.Normalize(
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5]
		)
	])
	to_pil = transforms.Compose([
		transforms.Normalize(
			mean=[-1.0, -1.0, -1.0],
			std=[2.0, 2.0, 2.0]
		),
		transforms.ToPILImage(),
	])
	
	# Create a background image to display the results
	bgc = Image.new('RGB', (128 * 11, 128 * 2), color='white')
	
	# Initialize the noise scheduler
	scheduler = CosineNoiseScheduler(n_timesteps=1000, device="cpu")
	
	# Preprocess the image
	x_0 = train_transformation(image)
	x_0 = x_0.unsqueeze(0)  # Add batch dimension
	
	# Paste the original image onto the background
	bgc.paste(to_pil(x_0[0].clamp(-1, 1)), (0, 0))
	
	# Perform the forward and backward diffusion processes
	for t in range(0, 1000, 100):
		t_batch = torch.full((1,), t, dtype=torch.long, device=scheduler.device)
		
		# Forward process: add noise
		x_t, epsilon = scheduler.q_forward(x_0, t_batch)
		bgc.paste(to_pil(x_t[0].clamp(-1, 1)), (128 + 128 * (t // 100), 0))
		
		# Backward process: denoise
		_, x_0_pred = scheduler.p_backward(x_t, epsilon, t_batch)
		bgc.paste(to_pil(x_0_pred[0].clamp(-1, 1)), (128 + 128 * (t // 100), 128))
	
	# Display the background image
	bgc.show()
	# Call the function to check noise recovery
	manual_noise = torch.randn_like(x_0)
	match, err = check_noise_prediction(scheduler, x_0, t_check=100, epsilon_manual=manual_noise)
