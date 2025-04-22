import yaml
import torch
from diffusion import UNet
import layers
from PIL import Image
from torchvision import transforms
import sys
import os

# Load the config file
def load_config(path="../../config/model/unet.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_image(image_path, size=(128, 128)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # batch dimension

# Initialize the model
model = UNet()

# Choose input: either dummy tensor or real image
use_image = True
image_path = "kitku.jpg"  # Set your image path here

if use_image and os.path.exists(image_path):
    image = load_image(image_path)  # Shape: (1, 3, H, W)
    print(f"Loaded image from {image_path} with shape {image.shape}")
else:
    image = torch.randn(1, 3, 128, 128)  # Dummy input
    print("Using dummy image.")

# Timestep tensor
time = torch.tensor([250], dtype=torch.long)

# Run the model
model.eval()
with torch.no_grad():
    output = model(image, time)

print(f"Output shape: {output.shape}")