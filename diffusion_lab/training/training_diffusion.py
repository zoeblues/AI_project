import os
from os.path import join as pjoin
import time

import hydra
import importlib
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import mlflow
import mlflow.pytorch
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset

log = logging.getLogger(__name__)


def train(model, scheduler, loader, optimizer, train_cfg, device):
	model = model.to(device)  # to make sure it's on an intended device
	criterion = nn.MSELoss()
	
	log.info("Starting training...")
	
	with mlflow.start_run(run_name=train_cfg.run_name):
		mlflow.log_param("training", train_cfg.name)
		mlflow.log_param("optimizer", train_cfg.optimizer.name)
		mlflow.log_params(train_cfg.optimizer.params)
		mlflow.log_params(train_cfg.params)
		
		for epoch in range(train_cfg.params.epochs):
			model.train()
			optimizer.zero_grad()
			
			start_time = time.time()
			
			running_loss = 0.0
			progress_bar = tqdm(loader, desc=f"Epoch: {epoch + 1}/{train_cfg.params.epochs}", unit=" bch")
			for i, batch in enumerate(progress_bar):
				batch = batch.to(device)
				
				timestep = torch.randint(1, train_cfg.params.timesteps, size=(batch.shape[0],), device=device)
				x_t, epsilon = scheduler.q_forward(batch, timestep)
				
				optimizer.zero_grad()
				out = model(x_t, timestep)
				loss = criterion(out, epsilon)
				running_loss += loss.item()
				# Normalize the loss for gradient accumulation
				loss = loss / train_cfg.params.accum_steps
				loss.backward()
				# Update the network only every N steps
				if (i + 1) % train_cfg.params.accum_steps == 0:
					optimizer.step()
					optimizer.zero_grad()
				
				progress_bar.set_postfix(loss=loss.item())
			
			avg_loss = running_loss / len(loader.dataset)
			elapsed = time.time() - start_time
			it_per_sec = (len(loader.dataset) + 1) / elapsed
			
			mlflow.log_metric("epoch_loss", running_loss, step=epoch)
			mlflow.log_metric("avg_loss", avg_loss, step=epoch)
			mlflow.log_metric("it_per_sec", it_per_sec, step=epoch)
			
			if epoch % train_cfg.params.log_step == 0:
				log.info(f"Epoch {epoch + 1}/{train_cfg.params.epochs} Loss: {avg_loss}")
			if epoch % train_cfg.params.save_step == 0 and epoch != 0:
				torch.save(model.state_dict(), pjoin(train_cfg.params.save_path, f"step-{epoch}.pth"))
	
	# Save model
	torch.save(model.state_dict(), pjoin(train_cfg.params.save_path, f"{train_cfg.params.model_name}.pth"))


@hydra.main(config_path="../../config", config_name="diffusion", version_base="1.3")
def main(cfg: DictConfig):
	mlflow.set_experiment(cfg.project.name)
	
	train_transform = transforms.Compose([
		transforms.Resize(128),  # todo: config
		transforms.RandomResizedCrop((128, 128), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),  # Convert image to tensor
		transforms.Normalize(  # Normalize RGB pixel values: [0, 255] -> [-1, 1]
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5]
		)
	])
	
	dataset = DiffusionDataset(dataset_path=cfg.train.params.dataset, transform=train_transform)
	loader = DataLoader(dataset, batch_size=cfg.train.params.bach_size, shuffle=True, **cfg.train.loader.params)
	
	# Loading model dynamically, based on config.
	model_path, model_name = cfg.model.type.rsplit(".", maxsplit=1)  # get module path, and class name
	model_cls = getattr(importlib.import_module(model_path), model_name)  # dynamic load a given class from a library
	model = model_cls(**cfg.model.params, device=cfg.train.params.device)  # create an instance, with params from config
	# Loading already trained model weights, if specified in the config (load_timestep != 0)
	if cfg.train.params.load_timestep != 0 and 'load_path' in cfg.train.params and 'load_timestep' in cfg.train.params:
		model.load_state_dict(torch.load(pjoin(cfg.train.params.load_path, f"{cfg.train.params.load_timestep}.pth")))
	
	# make sure the folder for train steps exists, create if it doesn't.
	os.makedirs(cfg.train.params.save_path, exist_ok=True)
	
	# Dynamic load of optimizer
	opti_path, opti_name = cfg.train.optimizer.type.rsplit(".", maxsplit=1)
	optimizer = getattr(importlib.import_module(opti_path), opti_name)(model.parameters(), **cfg.train.optimizer.params)
	
	# Dynamic load of noise scheduler
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler_cls = getattr(importlib.import_module(sche_path), sche_name)
	scheduler = scheduler_cls(**cfg.schedule.params, n_timesteps=cfg.train.params.timesteps, device=cfg.train.params.device)
	
	log.info("Options loaded successfully!")
	train(model, scheduler, loader, optimizer, cfg.train, device=cfg.train.params.device)


if __name__ == '__main__':
	main()
