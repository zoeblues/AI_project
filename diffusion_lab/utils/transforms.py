from torchvision import transforms


train_transform = transforms.Compose([
		transforms.Resize(64),
		transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),  # Convert image to tensor
		transforms.Normalize(  # Normalize RGB pixel values: [0, 255] -> [-1, 1]
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5]
		)
	])

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
