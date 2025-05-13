from torchvision import transforms

to_tensor = transforms.Compose([
	transforms.Resize(64),
	transforms.ToTensor(),
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
