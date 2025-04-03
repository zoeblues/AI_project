# AI_project
Diffusion model implementation for image generation for Artificial Intelligence subject at GUT.

## Setting up with make

Downloading base packages, requires make to be installed

```bash
make create
```

Alternatives, enter the command yourself:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data download

Downloading dataset and processing. Warning selected dataset as of now is 20GB

```bash
make download_data
```

Alternatives, enter the command yourself _(make sure the environment is active)_:

```bash
python scripts/download_animals_classification.py
```
