from __future__ import annotations

import copy
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens import HookedTransformerConfig
from transformer_lens.components import (
    Embed,
    PosEmbed,
    RMSNorm,
    TransformerBlock,
    Unembed,
    Attention,
    MLP
)
from transformer_lens.hook_points import HookedRootModule, HookPoint
from config import CaptchaConfig

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise conv: groups=dim, padding=3 to preserve size with kernel 7
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Pointwise convs implemented as Linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# Architecture 2:
class CNNEncoder(nn.Module):
    def __init__(self, cfg: CaptchaConfig):
        super().__init__()
        self.cfg = cfg
        
        # Conv2D 1: [7, 1, 16] (Height 70 -> 64)
        # Standard convolution for channel expansion (3 -> 16)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=1)
        
        # ConvNext Stage 1
        self.cnxt1 = nn.Sequential(
            ConvNextBlock(dim=16),
            ConvNextBlock(dim=16),
            ConvNextBlock(dim=16)
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D 2: [5, 1, 32] (Height 32 -> 28)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=1)
        
        # ConvNext Stage 2
        self.cnxt2 = nn.Sequential(
            ConvNextBlock(dim=32),
            ConvNextBlock(dim=32),
            ConvNextBlock(dim=32)
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.act = nn.GELU() 

    def forward(self, x: Float[torch.Tensor, "batch channel height width"]) -> Float[torch.Tensor, "batch tokens dim"]:
        # Expected input height: 70
        # x: [b, 1, 70, w]
        
        x = self.act(self.conv1(x)) # [b, 16, 64, w-6]
        x = self.cnxt1(x)           # [b, 16, 64, w-6]
        x = self.pool1(x)           # [b, 16, 32, (w-6)//2]
        
        x = self.act(self.conv2(x)) # [b, 32, 28, ...]
        x = self.cnxt2(x)           # [b, 32, 28, ...]
        x = self.pool2(x)           # [b, 32, 14, w_final]
        
        # Flatten pooling: extract 14x7 patches
        # token_height=14, token_width=7
        # num_tokens = width // 7
        
        # Unfold width to get patches of width 7
        # x shape: [b, 32, 14, w_feat]
        patches = x.unfold(3, 7, 7) # [b, 32, 14, num_tokens, 7]
        
        # Rearrange to [b, num_tokens, 32*14*7]
        # 32 channels * 14 height * 7 width = 3136
        patches = patches.permute(0, 3, 1, 2, 4).contiguous() 
        b, num_tokens, c, h, w_patch = patches.shape
        out = patches.view(b, num_tokens, c * h * w_patch)
        
        return out

'''
Architecture 1
class CNNEncoder(nn.Module):
    def __init__(self, cfg: CaptchaConfig):
        super().__init__()
        self.cfg = cfg
        # Conv2D 1: [7, 1, 16] (Height 70 -> 64)
        
        in_channels = 3 
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(7, 7), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D 2: [5, 1, 32] (Height 32 -> 28)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.act = nn.SiLU() # swish

    def forward(self, x: Float[torch.Tensor, "batch channel height width"]) -> Float[torch.Tensor, "batch tokens dim"]:
        # Expected input height: 70
        # x: [b, 1, 70, w]
        
        x = self.act(self.conv1(x)) # [b, 16, 64, w-6]
        x = self.pool1(x)           # [b, 16, 32, (w-6)//2]
        
        x = self.act(self.conv2(x)) # [b, 32, 28, ...]
        x = self.pool2(x)           # [b, 32, 14, w_final]
        
        # Flatten pooling: extract 14x7 patches
        # token_height=14, token_width=7
        # num_tokens = width // 7
        
        # Unfold width to get patches of width 7
        # x shape: [b, 32, 14, w_feat]
        patches = x.unfold(3, 7, 7) # [b, 32, 14, num_tokens, 7]
        
        # Rearrange to [b, num_tokens, 32*14*7]
        # 32 channels * 14 height * 7 width = 3136
        patches = patches.permute(0, 3, 1, 2, 4).contiguous() 
        b, num_tokens, c, h, w_patch = patches.shape
        out = patches.view(b, num_tokens, c * h * w_patch)
        
        return out
'''

class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block with Cross Attention.
    Structure: RMSNorm -> Self Attn -> RMSNorm -> Cross Attn -> RMSNorm -> MLP
    """
    def __init__(self, cfg: CaptchaConfig, block_index: int):
        super().__init__()
        self.cfg = cfg
        
        self.ln1 = RMSNorm(cfg)
        self_attn_cfg = copy.deepcopy(cfg)
        self_attn_cfg.attention_dir = 'bidirectional'
        self.self_attn = Attention(cfg, "global", block_index)
        
        self.ln2 = RMSNorm(cfg)
        # Cross attention needs to attend to all encoder outputs (bidirectional context)
        # We create a copy of config to force bidirectional attention for this layer component
        cross_cfg = copy.deepcopy(cfg)
        cross_cfg.attention_dir = "bidirectional"
        cross_cfg.positional_embedding_type = 'standard'
        self.cross_attn = Attention(cross_cfg, "global", block_index)
        
        self.ln3 = RMSNorm(cfg)
        self.mlp = MLP(cfg)
        
        # Hooks
        self.hook_self_attn_out = HookPoint()
        self.hook_cross_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_mid = HookPoint() # After self attn
        self.hook_resid_mid2 = HookPoint() # After cross attn
        self.hook_resid_post = HookPoint() # After MLP

    def forward(
        self, 
        x: Float[torch.Tensor, "batch pos d_model"], 
        encoder_out: Float[torch.Tensor, "batch enc_pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        # Self Attention
        resid = x
        x = self.ln1(x)
        # Self attention uses x for Q, K, V
        x = self.self_attn(x, x, x) 
        x = self.hook_self_attn_out(x)
        x = resid + x
        x = self.hook_resid_mid(x)
        
        # Cross Attention
        resid = x
        x = self.ln2(x)
        # Q from decoder (x), K, V from encoder_out
        x = self.cross_attn(x, encoder_out, encoder_out)
        x = self.hook_cross_attn_out(x)
        x = resid + x
        x = self.hook_resid_mid2(x)
        
        # MLP
        resid = x
        x = self.ln3(x)
        x = self.mlp(x)
        x = self.hook_mlp_out(x)
        x = resid + x
        x = self.hook_resid_post(x)
        
        return x

class CaptchaModel(HookedRootModule):
    def __init__(self, cfg: CaptchaConfig):
        super().__init__()
        self.cfg = cfg
        
        # --- Encoder ---
        self.cnn = CNNEncoder(cfg)
        
        # Projections: 3136 -> 392 -> 128 -> 64
        self.proj1 = nn.Linear(3136, 392)
        self.proj2 = nn.Linear(392, 128)
        self.proj3 = nn.Linear(128, 64)
        self.act = nn.SiLU()
        
        encoder_cfg = copy.deepcopy(cfg)
        encoder_cfg.attention_dir = 'bidirectional'

        # Transformer Encoder
        # Using standard TransformerBlock (self-attention only)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        # --- Decoder ---
        # DETR style: Learnable object queries instead of embeddings + pos enc
        # Shape: [1, seq_len, d_model] - broadcasted to batch
        self.decoder_queries = nn.Parameter(torch.randn(1, cfg.n_ctx, cfg.d_model))
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        
        self.decoder_final_ln = RMSNorm(cfg)
        self.unembed = Unembed(cfg)
        
        # Hooks setup
        self.setup()

    def forward(self, 
                image: Float[torch.Tensor, "batch 1 70 width"], 
                # decoder_input is no longer needed/used for queries
                decoder_input: Int[torch.Tensor, "batch seq_len"] = None):
        
        # --- Encoder Pass ---
        # CNN
        x = self.cnn(image)
        
        # Projections
        x = self.act(self.proj1(x))
        x = self.act(self.proj2(x))
        x = self.proj3(x) # [b, tokens, d_model]
        
        # Transformer Encoder
        # TransformerBlock expects: resid_pre. We can pass x directly.
        for block in self.encoder_blocks:
            x = block(x)
        encoder_out = x
        
        # --- Decoder Pass ---
        # Initialize decoder input with learned queries
        batch_size = image.shape[0]
        # Expand queries to batch size: [batch, seq_len, d_model]
        dec_x = self.decoder_queries.expand(batch_size, -1, -1)
            
        # Decoder Blocks with Cross Attention
        for block in self.decoder_blocks:
            dec_x = block(dec_x, encoder_out)
            
        dec_x = self.decoder_final_ln(dec_x)
        logits = self.unembed(dec_x)
        
        return logits