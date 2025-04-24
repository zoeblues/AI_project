"""
File taken from: https://github.com/mattroz/diffusion-ddpm/tree/main
"""
from diffusion_lab.models.layers import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionEmbedding, ConvUpBlock, \
	ResNetBlock
import torch
import torch.nn as nn
from omegaconf import DictConfig

# Load the config for unet


class UNet(nn.Module):
	def __init__(self, cfg: DictConfig):
		super().__init__()
		self.cfg=cfg
		
		# Load parameters from config
		base_channels = cfg.base_channels
		in_channels = cfg.in_channels
		out_channels = cfg.out_channels
		output_channels = cfg.output_channels
		num_att_heads = cfg.attention_heads
		num_groups = cfg.num_groups
		num_layers = cfg.num_layers
		time_emb_channels = base_channels * cfg.time_embedding_factor
		
		# Convert image into a base feature map
		self.initial_conv = nn.Conv2d(
			in_channels=in_channels, #3
			out_channels=out_channels, #64
			kernel_size=3,
			stride=1,
			padding=1
		)
		
		# Embed a time value into a feature vector
		self.positional_encoding = nn.Sequential(
			TransformerPositionEmbedding(dimension=base_channels),
			nn.Linear(base_channels, 4*base_channels),
			nn.GELU(),
			nn.Linear(4*base_channels, 4*base_channels)
		)
		
		# Reduce the spatial size (height/width) of the image
		# Increase the number of feature channels
		self.downsample_blocks = nn.ModuleList([
			ConvDownBlock(base_channels, base_channels,  num_layers, time_emb_channels, num_groups),
			ConvDownBlock(base_channels, base_channels,  num_layers, time_emb_channels, num_groups),
			ConvDownBlock(base_channels, base_channels*2,  num_layers, time_emb_channels, num_groups),
			AttentionDownBlock(base_channels*2, base_channels*2,  num_layers, time_emb_channels, num_groups, num_att_heads),
			ConvDownBlock(base_channels*2, base_channels*4,  num_layers, time_emb_channels, num_groups)
		])
		
		self.bottleneck = AttentionDownBlock(base_channels*4, base_channels*4, num_layers, time_emb_channels, num_groups, num_att_heads, downsample=False)
		
		# Upscale the features and merge them with the corresponding features from downsampling path
		self.upsample_blocks = nn.ModuleList([
			ConvUpBlock(base_channels*8, base_channels*4, num_layers, time_emb_channels, num_groups),
			AttentionUpBlock(base_channels*4 + base_channels*2, base_channels*2, num_layers, time_emb_channels, num_groups, num_att_heads),
			ConvUpBlock(base_channels*4, base_channels*2, num_layers, time_emb_channels, num_groups),
			ConvUpBlock(base_channels*2 + base_channels, base_channels, num_layers, time_emb_channels, num_groups),
			ConvUpBlock(base_channels*2, base_channels, num_layers, time_emb_channels, num_groups)
		])
		
		# Final image output, maps the last processed tensor back to desired num of channels
		self.output_conv = nn.Sequential(
			nn.GroupNorm(num_channels=base_channels, num_groups=num_groups),
			nn.SiLU(),
			nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)
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
