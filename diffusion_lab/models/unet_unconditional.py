import torch
import torch.nn as nn

from diffusion_lab.models.modules import *


class UNet(nn.Module):
	def __init__(self, in_channels, out_channels, base_channels=320, channel_multipliers=(1, 2, 4, 4),
	             emb_channels=256, norm_groups=8, dropout=0.2,
	             do_self_attention=(False, False, True, True), num_heads=4, device="cpu"):
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
				do_self_attention=do_self_attention[i], num_heads=num_heads,
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
				in_channels=prev_channels, out_channels=base_channels * multiply,
				skip_channels=base_channels * multiply,
				emb_channels=emb_channels, norm_groups=norm_groups, dropout=dropout,
				do_self_attention=do_self_attention[i], num_heads=num_heads,
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
