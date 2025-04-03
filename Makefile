
ENV_NAME = DiffusionProject
VENV_DIR := .venv
PYTHON_BASE := python3
PYTHON := .venv/bin/python


create:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON_BASE) -m venv $(VENV_DIR)
	@echo "Installing requirements..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS); \


remove:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)

run:
	$(PYTHON) file.py	

download_data:
	$(PYTHON) scripts/download_animals_classification.py
