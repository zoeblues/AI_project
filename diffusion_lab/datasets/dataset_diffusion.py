import torchvision.transforms
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# Default Transformations for image
_default_transform = transforms.Compose([
	transforms.Resize(128),
	transforms.ToTensor(),  # Convert image to tensor
	transforms.Normalize(  # Normalize RGB pixel values: [0, 255] -> [-1, 1]
		mean=[0.5, 0.5, 0.5],
		std=[0.5, 0.5, 0.5]
	)
])


class DiffusionDataset(Dataset):
	def __init__(self, dataset_path, transform=None):
		data = pd.read_csv(dataset_path)
		# Save the transformation provided by the user or use default
		self.transform = transform if transform is not None else _default_transform
		# Remember to change pandas.Series into a list or np.array for speed: `.tolist()` or `.to_numpy()`
		self.paths = data['path'].tolist()
		# One-Hot Encoding
		self.labels_lookup = sorted(data['label'].unique().tolist())
		self.labels = data['label'].apply(lambda x: self.labels_lookup.index(x)).to_numpy()
	
	def __len__(self):
		# Length of entire dataset is length of path column
		return len(self.paths)
	
	def __getitem__(self, idx):
		image = Image.open(self.paths[idx]).convert('RGB')  # read the image in RGB
		image = self.transform(image)
		
		return image


if __name__ == '__main__':
	path = 'data/resized_images/Cat/cat-test_(1).jpeg'
	image = Image.open(path).convert('RGB')
	
	train_transformation = transforms.Compose([
		# transforms.Resize(128),
		transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.2)),
		transforms.RandomHorizontalFlip(p=0.5),
		# transforms.ToTensor(),  # Convert image to tensor
		# transforms.Normalize(  # Normalize RGB pixel values: [0, 255] -> [-1, 1]
		# 	mean=[0.5, 0.5, 0.5],
		# 	std=[0.5, 0.5, 0.5]
		# )
	])
	
	img_size = 256
	n_examples = 4
	
	grid_img = Image.new('RGB', (img_size * (n_examples + 1), img_size), color='white')
	grid_img.paste(image, (0, 0))
	for i in range(n_examples):
		img = train_transformation(image)
		x = img_size * (i + 1)
		grid_img.paste(img, (x, 0))
	
	grid_img.show()

