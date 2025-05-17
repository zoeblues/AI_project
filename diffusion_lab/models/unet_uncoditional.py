import torch
import torch.nn as nn
import torch.nn.functional as F


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
		

class UNet(nn.Module):
	def __init__(self, in_channels, out_channels, base_channels=320, channel_multipliers=(1, 2, 4, 4),
	             emb_channels=256, norm_groups=8, dropout=0.2, device="cpu"):
		super().__init__()
		self.time_emb_channels = emb_channels
		self.device = device
		
		self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
		
		down_layers = []
		prev_channels = base_channels
		for i, multiply in enumerate(channel_multipliers):
			down_layers.append(ResolutionLayerDown(
				in_channels=prev_channels, out_channels=base_channels * multiply,
				emb_channels=emb_channels, norm_groups=norm_groups, dropout=dropout,
				do_self_attention=False  # todo: change
			))
			prev_channels = base_channels * multiply
		self.down_layers = nn.ModuleList(down_layers)
		
		self.middle_block = nn.ModuleList([
			ResBlock(prev_channels, prev_channels, emb_channels=emb_channels, norm_groups=norm_groups, dropout=dropout),
			ResBlock(prev_channels, prev_channels, emb_channels=emb_channels, norm_groups=norm_groups, dropout=dropout),
		])
		
		up_layers = []
		for i, multiply in enumerate(reversed(channel_multipliers)):
			up_layers.append(ResolutionLayerUp(
				in_channels=prev_channels, out_channels=base_channels * multiply, skip_channels=base_channels * multiply,
				emb_channels=emb_channels, norm_groups=norm_groups, dropout=dropout,
				do_self_attention=False  # todo: change
			))
			prev_channels = base_channels * multiply
		self.up_layers = nn.ModuleList(up_layers)
		
		self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
	
	def pos_encoding(self, t, channels):
		inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
		pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
		pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
		pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
		return pos_enc
	
	def forward(self, x, t):
		# Encode timesteps
		t = t.unsqueeze(-1).type(torch.float)
		t = self.pos_encoding(t, self.time_emb_channels)
		# Initial convolution to bring to base channels
		x = self.initial_conv(x)
		
		# Backlog for skip connections
		skip_connections = []
		# Down-sampling the image
		for layer in self.down_layers:
			x, skip = layer(x, t)
			skip_connections.append(skip)
		
		# Bottleneck
		for block in self.middle_block:
			x = block(x, t)
		
		# Reverse the skip connections
		skip_connections = list(reversed(skip_connections))
		# Up-sampling the features
		for layer, skip_x in zip(self.up_layers, skip_connections):
			x = layer(x, skip_x, t)
		
		# Return with output Conv
		return self.output_conv(x)


if __name__ == '__main__':
	model = UNet(in_channels=3, out_channels=3, channel_multipliers=(1, 2, 4), base_channels=32, device="mps").to("mps")
	model.eval()
	
	img = torch.randn(1, 3, 64, 64, device="mps")
	t = (torch.ones(1) * 999).long().to("mps")
	out = model(img, t)
