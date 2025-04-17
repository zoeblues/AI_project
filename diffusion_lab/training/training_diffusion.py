import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
import importlib

from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset


def train(model, scheduler, loader, optimizer, device):
	model = model.to(device)  # to make sure it's on intended device
	
	pass


@hydra.main(config_path="../../config", config_name="diffusion", version_base="1.3")
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))  # Print configuration
	
	dataset = DiffusionDataset(dataset_path=cfg.train.params.dataset)
	loader = DataLoader(dataset, batch_size=cfg.train.params.bach_size, shuffle=True)  # todo: config
	
	# Loading model dynamically, based on config.
	model_path, model_name = cfg.model.type.rsplit(".", maxsplit=1)  # obtain module path, and class name
	model_cls = getattr(importlib.import_module(model_path), model_name)  # dynamic load of given class from library
	model = model_cls(cfg.model.params)  # create instance of imported class, with parameters form config
	
	# Dynamic load of optimizer
	opti_path, opti_name = cfg.train.optimizer.type.rsplit(".", maxsplit=1)
	optimizer = getattr(importlib.import_module(opti_path), opti_name)(model.parameters(), **cfg.train.optimizer.params)
	
	# Dynamic load of noise scheduler
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler = getattr(importlib.import_module(sche_path), sche_name)(**cfg.schedule.params)
	
	train(model, scheduler, loader, optimizer, device=cfg.train.params.device)


if __name__ == '__main__':
	main()
