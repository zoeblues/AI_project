import torch
import torch.nn as nn


class SelfAttention(nn.Module):
	def __init__(self, channels, heads=4):
		super().__init__()
		self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
		self.ln1 = nn.LayerNorm(channels)
		self.ff = nn.Sequential(
			nn.LayerNorm(channels),
			nn.Linear(channels, channels),
			nn.GELU(),
			nn.Linear(channels, channels),
		)
	
	def forward(self, x):
		B, C, H, W = x.shape
		x = x.view(B, C, H * W).transpose(1, 2)
		
		# Self-attention with residual
		x_norm = self.ln1(x)
		attn_out, _ = self.mha(x_norm, x_norm, x_norm)
		x = x + attn_out
		
		# Feedforward with residual
		ff_out = self.ff(x)
		x = x + ff_out
		
		return x.transpose(1, 2).view(B, C, H, W)


class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None, emb_channels=256, norm_groups=8, dropout=0.2):
		super().__init__()
		# Option to add more controlled channel change, but defaults to out_channels
		if mid_channels is None:
			mid_channels = out_channels
		
		self.conv1 = nn.Sequential(
			nn.GroupNorm(norm_groups, in_channels),
			nn.SiLU(),
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
		)
		
		self.emb_projection = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_channels, mid_channels),
		)
		
		self.conv2 = nn.Sequential(
			nn.GroupNorm(norm_groups, mid_channels),
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
		h = h + self.emb_projection(t)[:, :, None, None]
		# Second: Group Normalization, Activation Function, Dropout, Convolution
		h = self.conv2(h)
		# Skip connection form the input with convolution output
		return self.skip(x) + h


class DownConv(nn.Module):
	def __init__(self):
		super().__init__()
		# Wrapper for MaxPool down sampling
		self.conv = nn.MaxPool2d(kernel_size=2, stride=2)
	
	def forward(self, x):
		return self.conv(x)


class UpConv(nn.Module):
	def __init__(self, out_channels):
		super().__init__()
		# Wrapper for Upsampling
		self.conv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
		)
	
	def forward(self, x):
		return self.conv(x)


class ResolutionLayerDown(nn.Module):
	def __init__(self, in_channels, out_channels, n_blocks=2,
	             emb_channels=256, norm_groups=8, dropout=0.2,
	             do_self_attention=False):
		super().__init__()
		# Add variable number of ResBlocks on a single Resolution Layer
		res_blocks = []
		for i in range(n_blocks):
			in_channels = in_channels if i == 0 else out_channels
			res_blocks.append(ResBlock(
				in_channels=in_channels,
				out_channels=out_channels,
				emb_channels=emb_channels,
				norm_groups=norm_groups,
				dropout=dropout,
			))
		self.res_blocks = nn.ModuleList(res_blocks)
		
		self.downsample = DownConv()
	
	def forward(self, x, t):
		# Sequentially apply all ResBlock over input
		for res_block in self.res_blocks:
			x = res_block(x, t)
		
		return self.downsample(x), x


class ResolutionLayerUp(nn.Module):
	def __init__(self, in_channels, out_channels, skip_channels, n_blocks=2,
	             emb_channels=256, norm_groups=8, dropout=0.2,
	             do_self_attention=False):
		super().__init__()
		self.upsample = UpConv(in_channels)
		
		res_blocks = []
		for i in range(n_blocks):
			in_channels = in_channels + skip_channels if i == 0 else out_channels
			res_blocks.append(ResBlock(
				in_channels=in_channels,
				out_channels=out_channels,
				emb_channels=emb_channels,
				norm_groups=norm_groups,
				dropout=dropout,
			))
		self.res_blocks = nn.ModuleList(res_blocks)
	
	def forward(self, x, skip_x, t):
		x = self.upsample(x)
		x = torch.cat([skip_x, x], dim=1)
		for res_block in self.res_blocks:
			x = res_block(x, t)
		
		return x
