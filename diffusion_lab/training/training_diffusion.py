import hydra
import importlib
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import mlflow
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset


log = logging.getLogger(__name__)


def train(model, scheduler, loader, optimizer, train_cfg, device):
	model = model.to(device)  # to make sure it's on an intended device
	criterion = nn.MSELoss()
	
	log.info("Starting training...")
	
	with mlflow.start_run(run_name=train_cfg.run_name):
		mlflow.log_params(train_cfg)
		
		for epoch in range(train_cfg.params.epochs):
			model.train()
			
			running_loss = 0.0
			progress_bar = tqdm(loader, desc=f"Epoch: {epoch+1}/{train_cfg.params.epochs}")
			for batch in progress_bar:
				
				timestep = torch.randint(1, train_cfg.params.timesteps, size=(batch.shape[0],), device=device)
				x_t, epsilon = scheduler.q_forward(batch, timestep)
				
				optimizer.zero_grad()
				out = model(x_t, timestep)
				loss = criterion(out, epsilon)
				loss.backward()
				optimizer.step()
				
				running_loss += loss.item()
				progress_bar.set_postfix(loss=loss.item())
			
			avg_loss = running_loss / len(loader)
			mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
	
	# Save model
	torch.save(model.state_dict(), train_cfg.model_path)


@hydra.main(config_path="../../config", config_name="diffusion", version_base="1.3")
def main(cfg: DictConfig):
	mlflow.set_tracking_uri("file:./mlruns")
	mlflow.set_experiment(cfg.project.name)
	
	dataset = DiffusionDataset(dataset_path=cfg.train.params.dataset)
	loader = DataLoader(dataset, batch_size=cfg.train.params.bach_size, shuffle=True)  # todo: config
	
	# Loading model dynamically, based on config.
	model_path, model_name = cfg.model.type.rsplit(".", maxsplit=1)  # get module path, and class name
	model_cls = getattr(importlib.import_module(model_path), model_name)  # dynamic load a given class from a library
	model = model_cls(**cfg.model.params)  # create instance of imported class, with parameters from config
	
	# Dynamic load of optimizer
	opti_path, opti_name = cfg.train.optimizer.type.rsplit(".", maxsplit=1)
	optimizer = getattr(importlib.import_module(opti_path), opti_name)(model.parameters(), **cfg.train.optimizer.params)
	
	# Dynamic load of noise scheduler
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler = getattr(importlib.import_module(sche_path), sche_name)(cfg.train.params.timesteps, **cfg.schedule.params)
	
	log.info("Options loaded successfully!")
	
	train(model, scheduler, loader, optimizer, cfg.train, device=cfg.train.params.device)


if __name__ == '__main__':
	main()
