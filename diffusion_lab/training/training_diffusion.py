import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset
from diffusion_lab.models.diffusion import UNet


def train(model, loader, optimizer, device):
	model = model.to(device)  # to make sure it's on intended device
	
	pass


def main():
	dataset = DiffusionDataset('data/diffusion_training.csv')  # todo: config
	loader = DataLoader(dataset, batch_size=64, shuffle=True)  # todo: config
	
	model = UNet(image_size=256)  # todo: cnf
	
	optimizer = Adam(model.parameters(), lr=1e-3)  # todo: config
	
	train(model, loader, optimizer, device="mps")


if __name__ == '__main__':
	main()
