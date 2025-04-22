"""
File taken from: https://github.com/mattroz/diffusion-ddpm/tree/main
"""
from layers import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionEmbedding, ConvUpBlock, \
	ResNetBlock
import torch
import torch.nn as nn
import yaml

# Load the config for unet
# todo change this into hydra format config
with open("../../config/model/unet.yaml", "r") as f:
	config = yaml.safe_load(f)["params"]


class UNet(nn.Module):
	def __init__(self):
		super().__init__()
		
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
			TransformerPositionEmbedding(dimension=64),
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
		
		# Upscale the features and merge them with the corresponding features from downsampling a path
		self.upsample_blocks = nn.ModuleList([
			ConvUpBlock(512, 256, 2, 256, 32),
			AttentionUpBlock(256 + 128, 128, 2, 256, 32, 4),
			ConvUpBlock(256, 128, 2, 256, 32),
			ConvUpBlock(128 + 64, 64, 2, 256, 32),
			ConvUpBlock(128, 64, 2, 256, 32)
		])
		
		# Final image output that maps the last processed tensor back to the desired num of channels
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
		skip_connections = list(reversed(skip_connections[:-1]))
		
		# Upsample path with skip connections
		for block, skip in zip(self.upsample_blocks, skip_connections):
			x = torch.cat([x, skip], dim=1)
			x = block(x, time_encoded)
		
		return self.output_conv(x)
