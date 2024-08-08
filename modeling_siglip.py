from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        hidden_size = 768,
        intermediate_size = 1072,
        num_hidden_layer = 12,
        num_attention_heads = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layer = num_hidden_layer
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig ):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size , channels, height, width] --> [Batch_size, Num_patches, Embd_Dim]
        return self.vision_model(pixel_values = pixel_values)

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size  # 16

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, # three i.e. RGB
            out_channels=self.embed_dim, # 768
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.floatTensor) -> torch.tensor:
        _, _, height, width = pixel_values.shape # [Batch_size , channels, height, width]
        # [Batch_size , channels, height, width] --> [Batch_size , embed_Dim, Num_Patches_H, Num_Patches_W]
        # Num_Patches_H = height // patch_size
        # Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_size , embed_Dim, Num_Patches_H, Num_Patches_W] --> [Batch_size , embed_Dim, Num_Patches]
        # Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_size , embed_Dim, Num_Patches] --> [Batch_size , Num_Patches, embed_Dim]
        embeddings = embeddings.transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings # [Batch_size , Num_Patches, embed_Dim]


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, input_embeds):
        input_embeds = input_embeds + self.layer_norm1(self.self_attn(input_embeds))
        input_embeds = input_embeds + self.layer_norm1(self.mlp(input_embeds))
        return input_embeds

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_Dim = config.hidden_size 

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_Dim, eps = config.layer_norm_eps)

    def forward(self, pixel_values):
        # pixel_values: [Batch_Size, Channels, Height, Width] --> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

    
