import torchvision.transforms
from torchvision import datasets, transforms
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
		data = pd.read_csv(dataset_path)[:1000]
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
