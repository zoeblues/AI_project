import hydra

from PIL import Image
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader

from diffusion_lab.utils.transforms import to_pil, train_transform
from diffusion_lab.datasets.dataset_diffusion import DiffusionDataset


def main(loader, b_size):
	n_cols = b_size
	n_rows = (len(loader.dataset) - 1) // n_cols + 1
	
	bgc = Image.new("RGB", (64 * n_cols, 64 * n_rows), color=(255, 255, 255))
	for row, batch in enumerate(loader):
		for col in range(batch.shape[0]):
			img = to_pil(batch[col])
			bgc.paste(img, (64 * col, 64 * row))
	
	bgc.show()


@hydra.main(config_path="../config", config_name="diffusion", version_base="1.3")
def load_run(cfg: DictConfig):
	dataset = DiffusionDataset(dataset_path=cfg.train.params.dataset, transform=train_transform)
	loader = DataLoader(dataset, batch_size=cfg.tests.params.ds_show_columns, shuffle=True, **cfg.train.loader.params)
	
	main(loader, b_size=cfg.tests.params.ds_show_columns)


if __name__ == '__main__':
	load_run()
