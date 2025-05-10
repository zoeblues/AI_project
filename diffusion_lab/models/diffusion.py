"""
UNet model for diffusion-based tasks.
Original source: https://github.com/mattroz/diffusion-ddpm
"""
import torch
import torch.nn as nn

from diffusion_lab.models.layers import (
	ConvDownBlock,
	AttentionDownBlock,
	AttentionUpBlock,
	TransformerPositionEmbedding,
	ConvUpBlock,
	ResNetBlock,
)


class UNet(nn.Module):
	def __init__(self, cfg, device='cpu'):
		super().__init__()
		self.cfg = cfg
		self.device = device
		
		# Initial convolution layer
		self.initial_conv = nn.Conv2d(
			in_channels=cfg.params.in_channels,
			out_channels=cfg.params.base_channels,
			kernel_size=3,
			stride=1,
			padding=1,
		)
		
		# Positional/time embedding
		self.positional_encoding = nn.Sequential(
			TransformerPositionEmbedding(
				dimension=cfg.params.base_channels,
				device=device
			),
			nn.Linear(cfg.params.base_channels, 4 * cfg.params.base_channels),
			nn.GELU(),
			nn.Linear(4 * cfg.params.base_channels, 4 * cfg.params.base_channels),
		)
		
		# Downsampling path
		self.downsample_blocks = nn.ModuleList([
			ConvDownBlock(
				cfg.params.base_channels,
				cfg.params.base_channels,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups
			),
			ConvDownBlock(
				cfg.params.base_channels,
				cfg.params.base_channels * 2,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups
			),
			AttentionDownBlock(
				cfg.params.base_channels * 2,
				cfg.params.base_channels * 2,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups,
				cfg.params.num_att_heads
			),
			ConvDownBlock(
				cfg.params.base_channels * 2,
				cfg.params.base_channels * 4,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups
			),
		])
		
		# Bottleneck
		self.bottleneck = AttentionDownBlock(
			cfg.params.base_channels * 4,
			cfg.params.base_channels * 4,
			cfg.params.num_layers,
			cfg.params.time_emb_channels,
			cfg.params.num_groups,
			cfg.params.num_att_heads,
			downsample=False
		)
		
		# Upsampling path
		self.upsample_blocks = nn.ModuleList([
			ConvUpBlock(
				cfg.params.base_channels * 4 + cfg.params.base_channels * 4,
				cfg.params.base_channels * 4,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups
			),
			AttentionUpBlock(
				cfg.params.base_channels * 4 + cfg.params.base_channels * 2,
				cfg.params.base_channels * 2,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups,
				cfg.params.num_att_heads
			),
			ConvUpBlock(
				cfg.params.base_channels * 2 + cfg.params.base_channels * 2,
				cfg.params.base_channels * 2,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups
			),
			ConvUpBlock(
				cfg.params.base_channels * 2 + cfg.params.base_channels,
				cfg.params.base_channels,
				cfg.params.num_layers,
				cfg.params.time_emb_channels,
				cfg.params.num_groups
			),
		])
		
		# Output projection
		self.output_conv = nn.Sequential(
			nn.GroupNorm(
				num_channels=cfg.params.base_channels,
				num_groups=cfg.params.num_groups
			),
			nn.SiLU(),
			nn.Conv2d(
				cfg.params.base_channels,
				cfg.params.in_channels,
				kernel_size=3,
				padding=1
			)
		)
	
	def forward(self, input_tensor, time):
		# Embed the time
		time_encoded = self.positional_encoding(time)
		
		# Initial convolution
		x = self.initial_conv(input_tensor)
		
		# Save inputs for skip connections
		skip_connections = [x]
		
		# Downsampling path
		for block in self.downsample_blocks:
			x = block(x, time_encoded)
			skip_connections.append(x)
		
		# Bottleneck
		x = self.bottleneck(x, time_encoded)
		
		# Reverse skip connections (remove bottleneck input)
		skip_connections = list(reversed(skip_connections))
		
		# Upsampling path with skip connections
		for block, skip in zip(self.upsample_blocks, skip_connections):
			x = torch.cat([x, skip], dim=1)
			x = block(x, time_encoded)
		
		return self.output_conv(x)
