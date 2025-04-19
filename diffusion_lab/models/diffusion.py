"""
File taken from: https://github.com/mattroz/diffusion-ddpm/tree/main
"""
from layers import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionEmbedding, ConvUpBlock, ResNetBlock
import torch
import torch.nn as nn
import yaml
# Load the config for unet
with open("prep_data.yml", "r") as f:
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

        self.initial_conv = nn.Conv2d(
            in_channels = input_channels,
            out_channels=base,
            kernel_size=3,
            stride=1,
            padding='same')


        self.positional_encoding = nn.Sequential(
            TransformerPositionEmbedding(dimension=base),
            nn.Linear(base, time_emb_channels),
            nn.GELU(),
            nn.Linear(time_emb_channels, time_emb_channels)
)
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(base, base, 2, num_groups, time_emb_channels),
            ConvDownBlock(base, base, 2, num_groups, time_emb_channels),
            ConvDownBlock(base, base * 2, 2, num_groups, time_emb_channels),
            AttentionDownBlock(base * 2, base * 2, 2, attention_heads, num_groups, time_emb_channels),
            ConvDownBlock(base * 2, base * 4, 2, num_groups, time_emb_channels)
        ])

        self.bottleneck = AttentionDownBlock(base * 4, base * 4, 2, attention_heads, num_groups, time_emb_channels, downsample=False)

        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(base * 8, base * 4, 2, num_groups, time_emb_channels),
            AttentionUpBlock(base * 4 + base * 2, base * 2, 2, attention_heads, num_groups, time_emb_channels),
            ConvUpBlock(base * 4, base * 2, 2, num_groups, time_emb_channels),
            ConvUpBlock(base * 2 + base, base, 2, num_groups, time_emb_channels),
            ConvUpBlock(base * 2, base, 2, num_groups, time_emb_channels)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=base * 2, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(base * 2, output_channels, 3, padding=1)
        )

    def forward(self, input_tensor, time):
        time_encoded = self.positional_encoding(time)

        initial_x = self.initial_conv(input_tensor)
        states_for_skip_connections = [initial_x]

        x = initial_x
        for block in self.downsample_blocks:
            x = block(x, time_encoded)
            states_for_skip_connections.append(x)

        states_for_skip_connections = list(reversed(states_for_skip_connections))
        x = self.bottleneck(x, time_encoded)

        for block, skip in zip(self.upsample_blocks, states_for_skip_connections):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)

        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        return self.output_conv(x)