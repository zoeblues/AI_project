"""
File taken from: https://github.com/mattroz/diffusion-ddpm/tree/main
"""
from layers import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionEmbedding, ConvUpBlock, ResNetBlock
import torch
import torch.nn as nn
import yaml
# Load the config for unet
with open("unet.yml", "r") as f:
    config=yaml.safe_load(f)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Set parameters according to config
        base = config["base_channels"]
        time_emb_channels = base * config["time_embedding_factor"]
        input_channels = config["input_channels"]
        num_groups = config["num_groups"]
        attention_heads = config["attention_heads"]
        output_channels = config["output_channels"]

        # Convert image into a base feature map
        self.initial_conv = nn.Conv2d(
            in_channels = input_channels,
            out_channels=base,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Embed a time value into a feature vector
        self.positional_encoding = nn.Sequential(
            TransformerPositionEmbedding(dimension=base),
            nn.Linear(base, time_emb_channels),
            nn.GELU(),
            nn.Linear(time_emb_channels, time_emb_channels)
        )

        # Reduce the spatial size (height/width) of the image
        # Increase the number of feature channels
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(base, base, 2, num_groups, time_emb_channels),
            ConvDownBlock(base, base, 2, num_groups, time_emb_channels),
            ConvDownBlock(base, base * 2, 2, num_groups, time_emb_channels),
            AttentionDownBlock(base * 2, base * 2, 2, attention_heads, num_groups, time_emb_channels),
            ConvDownBlock(base * 2, base * 4, 2, num_groups, time_emb_channels)
        ])

        self.bottleneck = AttentionDownBlock(base * 4, base * 4, 2, attention_heads, num_groups, time_emb_channels, downsample=False)

        # Upscale the features and merge them with the corresponding features from downsampling path
        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(base * 8, base * 4, 2, num_groups, time_emb_channels),
            AttentionUpBlock(base * 4 + base * 2, base * 2, 2, attention_heads, num_groups, time_emb_channels),
            ConvUpBlock(base * 4, base * 2, 2, num_groups, time_emb_channels),
            ConvUpBlock(base * 2 + base, base, 2, num_groups, time_emb_channels),
            ConvUpBlock(base * 2, base, 2, num_groups, time_emb_channels)
        ])

        # Final image output, that maps the last processed tensor back to desired num of channels
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=base * 2, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(base * 2, output_channels, 3, padding=1)
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