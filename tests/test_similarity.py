import importlib
import hydra
import numpy as np
import torch
import logging

from omegaconf import DictConfig

from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset
from diffusion_lab.sampling.sampling import sample_image
from diffusion_lab.utils.transforms import to_tensor, to_pil
from diffusion_lab.utils.plot_images import show_save_images
from diffusion_lab.utils.resolvers import *

log = logging.getLogger(__name__)


def main(model, scheduler, dataset, test_output_path, model_abs_path, device, **kwargs):
	model.load_state_dict(torch.load(model_abs_path, map_location=device))
	model.to(device)
	model.eval()
	
	sampled_img = to_pil(sample_image(model, scheduler)[0])
	img_vec = np.array(sampled_img, dtype=np.float32).flatten()
	
	highest_similarity = float("-inf")
	closest_img = None
	for img in dataset:
		dataset_img = to_pil(img)
		dataset_img_vec = np.array(dataset_img, dtype=np.float32).flatten()
		cos_similarity = np.dot(img_vec, dataset_img_vec) / (np.linalg.norm(img_vec) * np.linalg.norm(dataset_img_vec))
		if cos_similarity > highest_similarity:
			log.info(f"New highest similarity: {float(cos_similarity) * 100:.2f}%")
			highest_similarity = cos_similarity
			closest_img = dataset_img
	show_save_images([[sampled_img], [closest_img]], save_path=test_output_path)


@hydra.main(config_path="../config", config_name="diffusion", version_base="1.3")
def load_run(cfg: DictConfig):
	model_path, model_name = cfg.model.type.rsplit(".", maxsplit=1)
	model_cls = getattr(importlib.import_module(model_path), model_name)
	model = model_cls(**cfg.model.params)
	
	sche_path, sche_name = cfg.schedule.type.rsplit(".", maxsplit=1)
	scheduler_cls = getattr(importlib.import_module(sche_path), sche_name)
	scheduler = scheduler_cls(**cfg.schedule.params)
	
	dataset = DiffusionDataset(dataset_path=cfg.train.params.dataset, transform=to_tensor)
	
	main(model, scheduler, dataset, **cfg.tests.params)


if __name__ == '__main__':
	load_run()
