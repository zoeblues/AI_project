
ENV_NAME = DiffusionProject
ENV_FILE = environment.yaml

create:
	mamba env create -f $(ENV_FILE)

update:
	mamba env update -f $(ENV_FILE) --prune

clean:
	conda remove --name $(ENV_NAME) --all -y

run:
	conda run -n $(ENV_NAME) python file.py	


download_data:
	conda run -n $(ENV_NAME) python scripts/download_animals_classification.py
