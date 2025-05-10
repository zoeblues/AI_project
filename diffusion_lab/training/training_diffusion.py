import os
from os.path import join as pjoin
import time

import hydra
import importlib

import numpy as np
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import mlflow
import mlflow.pytorch
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset

log = logging.getLogger(__name__)


def warm_up_importance_sampling(model, scheduler, loader, optimizer, n_rounds, backlog_size, timesteps, device):
	criterion = nn.MSELoss(reduction='none')
	
	loss_backlog = torch.ones((timesteps, backlog_size), device=device)
	backlog_ptr = torch.zeros((timesteps,), device=device, dtype=torch.long)
	
	for epoch in range(n_rounds):
		model.train()
		
		progress_bar = tqdm(loader, desc=f"WarmUp: {epoch + 1}/{n_rounds}", unit=" bch")
		for i, batch in enumerate(progress_bar):
			batch = batch.to(device)
			
			t = torch.randint(1, timesteps, size=(batch.shape[0],), device=device)
			x_t, epsilon = scheduler.q_forward(batch, t)
			
			out = model(x_t, t)
			loss = criterion(out, epsilon).mean(dim=(1, 2, 3))  # (B,)
			
			# Backprop
			optimizer.zero_grad()
			loss.mean().backward()
			optimizer.step()
			
			loss_backlog[t, backlog_ptr[t]] = loss.detach()
			backlog_ptr[t] += 1
			backlog_ptr[t] %= backlog_size
			
			progress_bar.set_postfix(loss=loss.mean().detach().item())
	
	optimizer.zero_grad()
	return model, loss_backlog, backlog_ptr


def artifact_loss_chart(avg_loss, epoch, save_path="./"):
	os.makedirs(save_path, exist_ok=True)  # Good practise
	
	plt.figure(figsize=(18, 4))
	plt.bar(np.arange(len(avg_loss)), avg_loss, width=1.0)
	plt.xlabel("Timestep t")
	plt.ylabel("Average Loss")
	plt.title("Per-timestep Average Loss")
	plt.tight_layout()
	plt.grid(axis='y', linestyle='--', alpha=0.5)
	
	plt.savefig(pjoin(save_path, f"timestep_loss-{str(epoch)}.png"))
	plt.close()
	
	mlflow.log_artifact(pjoin(save_path, f"timestep_loss-{str(epoch)}.png"))
	pass


def train(model, scheduler, loader, optimizer, train_cfg, device='cpu'):
	model = model.to(device)  # to make sure it's on an intended device
	criterion = nn.MSELoss(reduction='none')
	
	log.info("Starting training...")
	
	importance_sampling = train_cfg.params.get("do-ImportanceSampling", False)
	timesteps = train_cfg.params.timesteps
	
	with mlflow.start_run(run_name=train_cfg.run_name):
		mlflow.log_param("training", train_cfg.name)
		mlflow.log_param("optimizer", train_cfg.optimizer.name)
		mlflow.log_param("importanceSampling", importance_sampling)
		mlflow.log_params(train_cfg.optimizer.params)
		mlflow.log_params(train_cfg.params)
		
		backlog_size = 10
		loss_backlog = torch.ones((timesteps, backlog_size), device=device)
		backlog_ptr = torch.zeros((timesteps,), device=device, dtype=torch.long)
		
		if importance_sampling:
			model, loss_backlog, backlog_ptr = warm_up_importance_sampling(model, scheduler, loader, optimizer, 5, backlog_size, timesteps, device)
			artifact_loss_chart(loss_backlog.mean(dim=1).cpu().numpy(), 0, save_path=train_cfg.params.artifact_path)
		
		for epoch in range(train_cfg.params.epochs):
			model.train()
			optimizer.zero_grad()
			
			running_loss = 0.0
			start_time = time.time()
			progress_bar = tqdm(loader, desc=f"Epoch: {epoch + 1}/{train_cfg.params.epochs}", unit=" bch")
			for i, batch in enumerate(progress_bar):
				batch = batch.to(device)
				
				weights = loss_backlog.mean(dim=1)
				t = torch.multinomial(weights, batch.shape[0], replacement=True).to(device)
				x_t, epsilon = scheduler.q_forward(batch, t)
				
				out = model(x_t, t)
				loss = criterion(out, epsilon).mean(dim=(1, 2, 3))  # (B,)
				running_loss += loss.detach().sum().item()
				
				# Importance Sampling Weights Update
				if importance_sampling:
					loss_backlog[t, backlog_ptr[t]] = loss.detach()
					backlog_ptr[t] += 1
					backlog_ptr[t] %= backlog_size
				
				# Normalize the loss for gradient accumulation
				loss = loss / train_cfg.params.accum_steps
				
				# Backpropagation
				loss.mean().backward()
				# Update the network only every N steps
				if (i + 1) % train_cfg.params.accum_steps == 0:
					optimizer.step()
					optimizer.zero_grad()
				
				progress_bar.set_postfix(loss=loss.mean().detach().item())
			
			avg_loss = running_loss / len(loader.dataset)
			elapsed = time.time() - start_time
			it_per_sec = (len(loader.dataset) + 1) / elapsed

			# ML Flow Logging
			mlflow.log_metric("avg_loss", avg_loss, step=epoch)
			mlflow.log_metric("items_per_sec", it_per_sec, step=epoch)
			mlflow.log_metric("elapsed", elapsed, step=epoch)
			if importance_sampling:
				for t_idx in [0, 200, 400, 600, 800, 999]:
					mlflow.log_metric(f"loss_t/{t_idx}", loss_backlog[t_idx].mean().item(), step=epoch)

			if (epoch + 1) % train_cfg.params.log_step == 0:
				log.info(f"Epoch {epoch + 1}/{train_cfg.params.epochs} Loss: {avg_loss}")
				if importance_sampling:
					artifact_loss_chart(loss_backlog.mean(dim=1).cpu().numpy(), epoch+1, save_path=train_cfg.params.artifact_path)
			if (epoch + 1) % train_cfg.params.save_step == 0 and epoch != 0:
				torch.save(model.state_dict(), pjoin(train_cfg.params.save_path, f"step-{epoch + 1}.pth"))
	
	# Save model
	torch.save(model.state_dict(), pjoin(train_cfg.params.save_path, f"{train_cfg.params.model_name}.pth"))


@hydra.main(config_path="../../config", config_name="diffusion", version_base="1.3")
def main(cfg: DictConfig):
	mlflow.set_experiment(cfg.project.name)
	
	train_transform = transforms.Compose([
		transforms.Resize(64),  # todo: config
		transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
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
	model = model_cls(cfg.model, device=cfg.train.params.device)  # create an instance, with params from config
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
