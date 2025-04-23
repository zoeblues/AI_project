"""
File taken from: https://github.com/mattroz/diffusion-ddpm/tree/main
"""
from diffusion_lab.models.layers import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionEmbedding, ConvUpBlock, \
	ResNetBlock
import torch
import torch.nn as nn
import yaml

# Load the config for unet
# todo change this into hydra format config
# with open("../../config/model/unet.yaml", "r") as f:
# 	config = yaml.safe_load(f)["params"]


class UNet(nn.Module):
	def __init__(self, device='cpu', **kwargs):
		super().__init__()
		
		# Remember the target device
		self.device = device
		
		# Convert image into a base feature map
		self.initial_conv = nn.Conv2d(
			in_channels=3,
			out_channels=64,
			kernel_size=3,
			stride=1,
			padding=1
		)
		
		# Embed a time value into a feature vector
		self.positional_encoding = nn.Sequential(
			TransformerPositionEmbedding(dimension=64, device=self.device),
			nn.Linear(64, 256),
			nn.GELU(),
			nn.Linear(256, 256)
		)
		
		# Reduce the spatial size (height/width) of the image
		# Increase the number of feature channels
		self.downsample_blocks = nn.ModuleList([
			ConvDownBlock(64, 64, 2, 256, 32),
			ConvDownBlock(64, 64, 2, 256, 32),
			ConvDownBlock(64, 128, 2, 256, 32),
			AttentionDownBlock(128, 128, 2, 256, 32, 4),
			ConvDownBlock(128, 256, 2, 256, 32)
		])
		
		self.bottleneck = AttentionDownBlock(256, 256, 2, 256, 32, 4, downsample=False)
		
		# Upscale the features and merge them with the corresponding features from downsampling path
		self.upsample_blocks = nn.ModuleList([
			ConvUpBlock(512, 256, 2, 256, 32),
			AttentionUpBlock(256 + 128, 128, 2, 256, 32, 4),
			ConvUpBlock(256, 128, 2, 256, 32),
			ConvUpBlock(128 + 64, 64, 2, 256, 32),
			ConvUpBlock(128, 64, 2, 256, 32)
		])
		
		# Final image output, that maps the last processed tensor back to desired num of channels
		self.output_conv = nn.Sequential(
			nn.GroupNorm(num_channels=64, num_groups=32),
			nn.SiLU(),
			nn.Conv2d(64, 3, 3, padding=1)
		)
	
	def forward(self, input_tensor, time):
		# Embed the time
		time_encoded = self.positional_encoding(time)
		# Apply initial conv to image
		x = self.initial_conv(input_tensor)
		
		# Save for skip connections - prevent network from forgetting input structure
		skip_connections = [x]
		
		# Downsample path, save each output for skip connections
		for block in self.downsample_blocks:
			x = block(x, time_encoded)
			skip_connections.append(x)
		
		x = self.bottleneck(x, time_encoded)
		
		# Reverse skip list and remove the last one (bottleneck)
		skip_connections = list(reversed(skip_connections))
		
		# Upsample path with skip connections
		for block, skip in zip(self.upsample_blocks, skip_connections):
			x = torch.cat([x, skip], dim=1)
			x = block(x, time_encoded)
		
		return self.output_conv(x)


if __name__ == '__main__':
	# Please run with working dir as the main project dir
	import os
	from PIL import Image
	from torchvision import transforms
	
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
	image_path = "data/resized_images/Cat/cat-test_(1).jpeg"  # Set your image path here
	
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
