"""
File taken from: https://github.com/mattroz/diffusion-ddpm/tree/main
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Load from config
def load_config(path="config/train/prep_data.yml"): #todo change path so its correct
    with open(path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
'''
from paper "Attention is all you need"
Based on Appendix B from "DDPM" paper
 - 32x32 models use four feature map resolution (32x32 to 4x4)
 - 256x256 models use six such maps
 - all models have two convolutional residual blocks per resolution level
 - all models have 16x16 resolution between the convolutional blocks
 - diffusion time t is specified by Transformer sinusoidal position embedding
'''

# Generate sinusoidal time embeddings
class TransformerPositionEmbedding(nn.Module):
    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        even_indices = torch.arange(0, self.dimension,2)
        log_term = torch.log(torch.tensor(10000.0))/ self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        return self.pe_matrix[timestep].to(timestep.device)

# Based on a ddpm paper we take group normalization instead of weight
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_ch, groups=8):
        super().__init__()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_ch, out_channels))
        self.block1 = ConvBlock(in_channels, out_channels, groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        if self.time_proj:
            t_emb = self.time_proj(t_emb)[:, :, None, None]
            h = h + t_emb
        h = self.block2(h)
        return h + self.res_conv(x)

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)

    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0):
        super(UpSampleBlock, self).__init__()

        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels,out_channels,3,padding=1)

    def forward(self, input_tensor):
        x = F.interpolate(input_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True)
        x = self.conv(x)
        return x

class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, downsample=True):
       super(ConvDownBlock, self).__init__()
       resnet_blocks = []
       for i in range(num_layers):
           in_channels = in_channels if i==0 else out_channels
           resnet_block = ResNetBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      time_emb_channels=time_emb_channels,
                                      num_groups=num_groups)
           resnet_blocks.append(resnet_block)
       self.resnet_blocks = nn.ModuleList(resnet_blocks)

       self.downsample = DownSampleBlock(in_channels=out_channels, out_channels=in_channels, stride=2,padding=1) \
       if downsample \
       else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.downsample:
            x = self.downsample(x)
        return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_channels, num_groups, upsample=True):
        super(ConvUpBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       time_emb_channels=time_emb_channels,
                                       num_groups=num_groups)
            resnet_blocks.append(resnet_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        self.upsample = UpSampleBlock(in_channels=out_channels, out_channels=out_channels) \
            if upsample \
            else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.upsample:
            x = self.upsample(x)
        return x

 # SelfAttentionBlocks in original paper they are applied on 16x16, here it is 32x32
class SelfAttentionBlock(nn.Module):
   def __init__(self, num_heads, in_channels, num_groups=32, embedding_dim=256): #todo into config
       super(SelfAttentionBlock, self).__init__()
       self.num_heads = num_heads
       self.d_model = embedding_dim
       self.d_keys = embedding_dim // num_heads
       self.d_values = embedding_dim // num_heads

       self.query_projection = nn.Linear(in_channels, embedding_dim)
       self.key_projection = nn.Linear(in_channels, embedding_dim)
       self.value_projection = nn.Linear(in_channels, embedding_dim)

       self.final_projection = nn.Linear(embedding_dim, embedding_dim)
       self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

   def split_features_for_heads(self, tensor):
       # We receive Q, K and V at shape [batch, h*w, embedding_dim].
       # This method splits embedding_dim into 'num_heads' features so that
       # each channel becomes of size embedding_dim / num_heads.
       # Output shape becomes [batch, num_heads, h*w, embedding_dim/num_heads],
       # where 'embedding_dim/num_heads' is equal to d_k = d_k = d_v = sizes for
       # K, Q and V respectively, according to paper.
       batch, hw, emb_dim = tensor.shape
       channels_per_head = emb_dim // self.num_heads
       heads_splitted_tensor = torch.split(tensor, split_size_or_sections=channels_per_head, dim=-1)
       heads_splitted_tensor = torch.stack(heads_splitted_tensor, 1)
       return heads_splitted_tensor

   def forward(self, input_tensor):
       x = input_tensor
       batch, features, h, w = x.shape
       # Do reshape and transpose input tensor since we want to process depth feature maps, not spatial maps
       x = x.view(batch, features, h * w).transpose(1, 2)

       # Get linear projections of K, Q and V according to Fig. 2 in the original Transformer paper
       queries = self.query_projection(x)  # [b, in_channels, embedding_dim]
       keys = self.key_projection(x)  # [b, in_channels, embedding_dim]
       values = self.value_projection(x)  # [b, in_channels, embedding_dim]

       # Split Q, K, V between attention heads to process them simultaneously
       queries = self.split_features_for_heads(queries)
       keys = self.split_features_for_heads(keys)
       values = self.split_features_for_heads(values)

       # Perform Scaled Dot-Product Attention (eq. 1 in the Transformer paper).
       # Each SDPA block yields tensor of size d_v = embedding_dim/num_heads.
       scale = self.d_keys ** -0.5
       attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
       attention_scores = torch.matmul(attention_scores, values)

       # Permute computed attention scores such that
       # [batch, num_heads, h*w, embedding_dim] --> [batch, h*w, num_heads, d_v]
       attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

       # Concatenate scores per head into one tensor so that
       # [batch, h*w, num_heads, d_v] --> [batch, h*w, num_heads*d_v]
       concatenated_heads_attention_scores = attention_scores.view(batch, h * w, self.d_model)

       # Perform linear projection and reshape tensor such that
       # [batch, h*w, d_model] --> [batch, d_model, h*w] -> [batch, d_model, h, w]
       linear_projection = self.final_projection(concatenated_heads_attention_scores)
       linear_projection = linear_projection.transpose(-1, -2).reshape(batch, self.d_model, h, w)

       # Residual connection + norm
       x = self.norm(linear_projection + input_tensor)
       return x

class AttentionDownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 time_emb_channels,
                 num_groups,
                 num_att_heads,
                 downsample=True):
        """
        AttentionDownBlock consists of ResNet blocks with Self-Attention blocks in-between
        :param in_channels:
        :param out_channels:
        :param num_layers:
        :param time_emb_channels:
        :param num_groups:
        :param num_att_heads:
        :param downsample:
        """
        super(AttentionDownBlock, self).__init__()

        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       time_emb_channels=time_emb_channels,
                                       num_groups=num_groups)
            attention_block = SelfAttentionBlock(in_channels=out_channels,
                                                 embedding_dim=out_channels,
                                                 num_heads=num_att_heads,
                                                 num_groups=num_groups)

            resnet_blocks.append(resnet_block)
            attention_blocks.append(attention_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.downsample = DownSampleBlock(in_channels=out_channels, out_channels=out_channels, stride=2, padding=1) \
            if downsample \
            else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class AttentionUpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 time_emb_channels,
                 num_groups,
                 num_att_heads,
                 upsample=True):
        """
        :param in_channels:
        :param out_channels:
        :param num_layers:
        :param time_emb_channels:
        :param num_groups:
        :param num_att_heads:
        :param upsample:
        """
        super(AttentionUpBlock, self).__init__()

        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       time_emb_channels=time_emb_channels,
                                       num_groups=num_groups)
            attention_block = SelfAttentionBlock(in_channels=out_channels,
                                                 embedding_dim=out_channels,
                                                 num_heads=num_att_heads,
                                                 num_groups=num_groups)

            resnet_blocks.append(resnet_block)
            attention_blocks.append(attention_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.upsample = UpSampleBlock(in_channels=out_channels, out_channels=out_channels) \
            if upsample \
            else None

    def forward(self, input_tensor, time_embedding):
        x = input_tensor
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.upsample:
            x = self.upsample(x)
        return x
