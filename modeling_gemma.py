from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.nn import functional as F 
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionModel, SiglipVisionConfig

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # shape of KV cache : [Bsz, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM]
            return self.key_cache[0].shape(-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads, # number of heads for Query
        num_key_value_heads, # number of heads for Value and Key
        head_dim=256, # how many dimension each head will watch
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.projector = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias = True)

    def forward(self, x):
        return self.projector(x)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.tensor:
    return  hidden_states.repeat_interleave(n_rep, dim=1)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # 8
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads # 1
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0            
        """
        Number of heads = 8
        Hidden_size = 1024
        Head_Dim = 1024 // 8 == 128
        Wq: [1024, 8 * 128] == [1024, 1024]

        Wk: [1024, 1 * 128]
        Wv: [1024, 1 * 128]
        """
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias) # [1024, 1024]

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embedding = self.max_position_embeddings,
            base = self.rope_theta
        )

    def forward(self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
             bsz, q_len, _ = hidden_states.size() # []

             query_states = self.q_proj(hidden_states)

             key_states = self.k_proj(hidden_states)
             value_states = self.v_proj(hidden_states)

             query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

             key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
             value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

             cos, sin = self.rotary_emb(value_states, position_ids, seq_len = None)

             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

             if kv_cache is not None:
                key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

             # repeat the key and values to match the number of heads of the query
             key_states = repeat_kv(key_states, self.num_key_value_groups)
             value_states = repeat_kv(query_states, self.num_key_value_groups)

             attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))/ math.sqrt(self.head_dim)

             assert attention_mask is not None
             attn_weights = attn_weights * attention_mask

             attn_weights = F.softmax(attn_weights, dim = -1, dtype=torch.float32).to(query_states.dtype)
             attn_weights = F.dropout(attn_weights, p = self.attention_dropout, training = self.training)

             attn_output = torch.matmul(attn_weights, value_states)

             attn_output = attn_output.transpose(1, 2).contiguous()
             attn_output = attn_output.view(bsz, q_len, -1) # [Bsz, Seq_len, Num_head_Q * Head_Dim]

             attn_output = self.o_proj(attn_output)

             return attn_output, attn_weights

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            # hidden_states : [Batch_size, seq_len, Hidden_size]
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _, = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_mask=position_ids,
                kv_cache=kv_cache
            )
            hidden_states += residual

            residual = hidden_states
            hidden_states = self.post_attn_norm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states += residual

            return hidden_states # [Batch_size, seq_len, Hidden_size]

class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in config.num_hidden_layers]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    )-> torch.FloatTensor:
        hidden_states = inputs_embeds

        normalizer = torch.tensor(config.hidden_size ** -0.5, dtype= inputs_embeds.dtype)
        hidden_states *= normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
         ) -> Tuple:
            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                kv_cache=kv_cache,
            )

            hidden_states = outputs
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            return_data = {
                "logits": logits,
            }

            if kv_cache is not None:
                # Return the updated cache
                return_data["kv_cache"] = kv_cache

            return return_data

class PaliGemmaForConditionalGeneration(nn.Module):
    def __int__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config 
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        """
         tokens initially --> [560, 560, 560, 560, 1, 56 , 789, 43, 23, 45, 27, 2]
         text_mask --> [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
         image_mask --> [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
         pad_mask --> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        # Shape: [Batch_Size, Seq_Len]. True for text tokens, image tokens, padding tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ##  ----------- Create The Attention Mask -------------  ##

        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, bcoz we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.fill(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.fill(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_size, Q_len, KV_len] --> [Batch_size, Num_Heads ,Q_len, KV_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            
        else:
            # create a position_ids based on the size of the attention mask
            # for masked tokens, use the number 1 as positions
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None, # from paliGemma processor, structure : image_tokens...bos...prefix_tokens
        pixel_values: torch.FloatTensor = None, # pixel_values from PaliGemma processor
        attention_mask: Optional[torch.tensor] = None, 
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra input embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] --> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # [Batch_Size, Num_Patches, Embed_Dim] --> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs

