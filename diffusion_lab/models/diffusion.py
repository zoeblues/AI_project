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
	def __init__(self, cfg: DictConfig, device='cpu'):
		super().__init__()
		self.cfg=cfg
		
		# Load parameters from config
		base_channels = cfg.base_channels
		in_channels = cfg.in_channels
		num_att_heads = cfg.attention_heads
		num_groups = cfg.num_groups
		num_layers = cfg.num_layers
		time_emb_channels = base_channels * cfg.time_embedding_factor
		
		# Convert image into a base feature map
		self.initial_conv = nn.Conv2d(
			in_channels=in_channels,
			out_channels=base_channels,
			kernel_size=3,
			stride=1,
			padding=1
		)
		
		# Embed a time value into a feature vector
		self.positional_encoding = nn.Sequential(
			TransformerPositionEmbedding(dimension=base_channels, device=device),
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
			nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
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
	
	cfg = DictConfig({
		'base_channels': 64,
		'image_size': 128,
		'in_channels': 3,
		'num_groups': 32,
		'time_embedding_factor': 4,
		'attention_heads': 4,
		'out_channels': 3,
		'num_layers': 2
	})
	
	# Initialize the model
	model = UNet(cfg, device='cpu')
	
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
