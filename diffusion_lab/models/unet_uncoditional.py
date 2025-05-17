import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None, embed_channels=256, norm_groups=8, dropout=0.2):
		super().__init__()
		# Option to add more controlled channel change, but defaults to out_channels
		if mid_channels is None:
			mid_channels = out_channels
		
		self.conv1 = nn.Sequential(
			nn.GroupNorm(norm_groups, in_channels),
			nn.SiLU(),
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
		)
		
		self.embed_projection = nn.Sequential(
			nn.SiLU(),
			nn.Linear(embed_channels, mid_channels),
		)
		
		self.conv2 = nn.Sequential(
			nn.GroupNorm(norm_groups, in_channels),
			nn.SiLU(),
			nn.Dropout2d(dropout),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
		)
		
		# The conditioning is here to make sure addition is possible.
		# If `in_channels != out_channels` addition is not possible, so they need to be transformed,
		# Then we use the simplest Conv with kernel=1 to change no. channels
		self.skip = (
			nn.Identity()
			if in_channels == out_channels
			else nn.Conv2d(in_channels, out_channels, kernel_size=1)
		)
	
	def forward(self, x, t):
		# First: Group Normalization, Activation Function, Convolution
		h = self.conv1(x)
		# Add Timestep embedding
		h = h + self.embed_projection(t)[:, :, None, None]
		# Second: Group Normalization, Activation Function, Dropout, Convolution
		h = self.conv2(h)
		# Skip connection form the input with convolution output
		return self.skip(x) + h


class ResolutionLayerDown(nn.Module):
	def __init__(self, in_channels, out_channels,
	             embed_channels=256, norm_groups=8, dropout=0.2,
	             n_blocks=2, init_downsample=True, do_self_attention=False):
		super().__init__()
		# For ease of use whit UNet skip-connections, downsampling is done at the beginning of Resolution layer.
		# However not all Layers start with down sampling, example first Layer.
		self.downsample = nn.MaxPool2d(2, stride=2) if init_downsample else None
		
		# Add variable number of ResBlocks on a single Resolution Layer
		res_blocks = []
		for i in range(n_blocks):
			in_channels = in_channels if i == 0 else out_channels
			res_blocks.append(ResBlock(
				in_channels=in_channels,
				out_channels=out_channels,
				embed_channels=embed_channels,
				norm_groups=norm_groups,
				dropout=dropout,
			))
		self.res_blocks = nn.ModuleList(res_blocks)

	def forward(self, x, t):
		# Only subsample when specified in `__init__`
		if self.downsample is not None:
			x = self.downsample(x)
		# Sequentially apply all ResBlock over input
		for res_block in self.res_blocks:
			x = res_block(x, t)
		
		return x
		
		

class UNet(nn.Module):
	def __init__(self, in_channels, out_channels, base_channels=32, channel_multipliers=(1, 2, 4, 8)):
		super().__init__()
