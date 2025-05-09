import torch
from PIL import Image
from torchvision import transforms

from diffusion_lab.models.noise_scheduler import NoiseScheduler, CosineNoiseScheduler


to_tensor = transforms.Compose([
	transforms.Resize(64),
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


def test_schedule(scheduler: NoiseScheduler, image, t):
	x_0 = to_tensor(image)
	x_0 = x_0.unsqueeze(0)
	
	torch.manual_seed(0)
	noise = torch.randn(x_0.shape, device=scheduler.device)
	
	x_t, _ = scheduler.q_forward(x_0, torch.tensor([t]), epsilon=noise)
	x_prev, x_0_pred = scheduler.p_backward(x_t, noise, torch.tensor([t]))
	
	return to_pil(x_t[0]), to_pil(x_prev[0]), to_pil(x_0_pred[0])


def main():
	# DO NOT CHANGE!
	path = 'data/resized_images/Cat/cat-test_(1).jpeg'
	image = Image.open(path).convert('RGB')
	
	scheduler = CosineNoiseScheduler(n_timesteps=1000)
	test_timesteps = [1, 50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 999]
	
	bgc = Image.new('RGB', (64 * len(test_timesteps), 64 * 2), color='white')
	
	for i, t in enumerate(test_timesteps):
		noised_img, _, x_0_approx = test_schedule(scheduler, image, t)
		
		bgc.paste(noised_img, (64 * i, 0))
		bgc.paste(x_0_approx, (64 * i, 64))
		# bgc.paste(to_pil(x_prev[0]), (64 * i, 128))
	
	bgc.show()


if __name__ == '__main__':
	# !!! Remember to set your working dir to the main project dir
	main()


