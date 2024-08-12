from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size = 768,
        intermediate_size = 1072,
        num_hidden_layers = 12,
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
        self.num_hidden_layers = num_hidden_layers
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

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size 

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)

    def forward(self, pixel_values):
        # pixel_values: [Batch_Size, Channels, Height, Width] --> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

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

    def forward(self, pixel_values: torch.FloatTensor) -> torch.tensor:
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

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.tensor) -> Tuple[torch.tensor, Optional[torch.tensor]]:
        batch_size , seq_len, _ = hidden_states.size()
        # Q, K, V : [batch_size , Num_patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Now, break down Q, K, V into smaller tokens for each heads
        # [batch_size , Num_patches, Embed_Dim] --> [batch_size , Num_patches, Num_heads, Head_Dim] --> [batch_size , Num_heads, Num_patches, Head_Dim]
        query_states = query_states.view(batch_size , seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size , seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size , seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale)
        # atten_weights: [batch_size , Num_heads, Num_patches, Num_patches]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states) # [batch_size , Num_heads, Num_patches, Head_Dim]
        # [batch_size , Num_heads, Num_patches, Head_Dim] --> [batch_size , Num_patches, Num_heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size , Num_patches, Num_heads, Head_Dim] --> [batch_size , Num_patches, embed_dim]
        # concat tokens from all 8 heads
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        # [batch_size , Num_patches, embed_dim] --> [batch_size , Num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.fc2(nn.functional.gelu(self.fc1(hidden_states), approximate="tanh"))


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states):
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states 

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states 
        return hidden_states

