"""
Sequence model implementations with registration.
All sequence models inherit from BaseSequenceModel.
"""

import copy
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Optional, TYPE_CHECKING, Any

from transformer_lens.components import (
    TransformerBlock,
    RMSNorm,
    Attention,
    MLP
)
from transformer_lens.hook_points import HookPoint

from .base import BaseSequenceModel
from ..registry import REGISTRY

from .base import BaseSequenceModel
from ..registry import REGISTRY
from ...config import (
    TransformerEncoderConfig, 
    TransformerDecoderConfig, 
    RNNConfig, 
    BiLSTMConfig
)

Tensor = torch.Tensor


class CaptchaDecoderBlock(nn.Module):
    """
    Transformer Decoder Block with Cross Attention.
    Structure: RMSNorm -> Self Attn -> RMSNorm -> Cross Attn -> RMSNorm -> MLP
    """
    def __init__(self, cfg: TransformerDecoderConfig, block_index: int):
        super().__init__()
        self.cfg = cfg
        
        self.ln1 = RMSNorm(cfg)
        
        # Self attention uses bidirectional context for the queries themselves
        self_attn_cfg = copy.deepcopy(cfg)
        self_attn_cfg.attention_dir = 'bidirectional'
        self.self_attn = Attention(self_attn_cfg, "global", block_index)
        
        self.ln2 = RMSNorm(cfg)
        
        # Cross attention attends to encoder outputs
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
        self.hook_resid_mid = HookPoint()   # After self attn
        self.hook_resid_mid2 = HookPoint()  # After cross attn
        self.hook_resid_post = HookPoint()  # After MLP
        
    def forward(
        self, 
        x: Float[torch.Tensor, "batch pos d_model"], 
        encoder_out: Float[torch.Tensor, "batch enc_pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        # Self Attention
        resid = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x) 
        x = self.hook_self_attn_out(x)
        x = resid + x
        x = self.hook_resid_mid(x)
        
        # Cross Attention
        resid = x
        x = self.ln2(x)
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


class TransformerEncoderModel(BaseSequenceModel):
    """
    Standard Transformer Encoder stack (Self-Attention only).
    """
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg
        
        # Ensure bidirectional attention for visual/sequence encoding
        enc_cfg = copy.deepcopy(cfg)
        enc_cfg.attention_dir = 'bidirectional'
        
        self.blocks = nn.ModuleList([
            TransformerBlock(enc_cfg, i) for i in range(cfg.n_layers)
        ])

    @property
    def requires_cross_attention(self) -> bool:
        return False

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        for block in self.blocks:
            x = block(x)
        return x


class TransformerDecoderModel(BaseSequenceModel):
    """
    Transformer Decoder stack (Self-Attention + Cross-Attention).
    Used for DETR-style decoding where 'x' are queries and 'encoder_out' is visual features.
    """
    def __init__(self, cfg: TransformerDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            CaptchaDecoderBlock(cfg, i) for i in range(cfg.n_layers)
        ])

    @property
    def requires_cross_attention(self) -> bool:
        return True

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        
        if encoder_out is None:
            raise ValueError("TransformerDecoderModel requires encoder_out for cross-attention")
            
        for block in self.blocks:
            x = block(x, encoder_out)
        return x


class RNNSequenceModel(BaseSequenceModel):
    """
    Standard RNN Sequence Model.
    """
    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg
        
        self.rnn = nn.RNN(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=0.1 if cfg.n_layers > 1 else 0.0,
            bidirectional=False # Standard RNN
        )
        self.act = nn.ReLU() # Often helpful after RNN

    @property
    def requires_cross_attention(self) -> bool:
        return False

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        
        # RNN returns (output, h_n)
        out, _ = self.rnn(x)
        return out


class BiLSTMSequenceModel(BaseSequenceModel):
    """
    Bidirectional LSTM Sequence Model.
    Projects output back to d_model size.
    """
    def __init__(self, cfg: BiLSTMConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.d_model % 2 != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be even for BiLSTM (hidden_size = d_model // 2)")
            
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model // 2, # Halve hidden size so bidirectional concat matches d_model
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if cfg.n_layers > 1 else 0.0
        )

    @property
    def requires_cross_attention(self) -> bool:
        return False

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        
        out, _ = self.lstm(x)
        return out


# ============================================================================
# REGISTRATION
# ============================================================================

REGISTRY.register_sequence_model(
    name="transformer_encoder",
    cls=TransformerEncoderModel,
    description="Stack of Transformer Blocks (Self-Attention only)",
    requires_cross_attention=False
)

REGISTRY.register_sequence_model(
    name="transformer_decoder",
    cls=TransformerDecoderModel,
    description="Stack of Transformer Decoder Blocks (Self + Cross Attention)",
    requires_cross_attention=True
)

REGISTRY.register_sequence_model(
    name="rnn",
    cls=RNNSequenceModel,
    description="Standard Recurrent Neural Network",
    requires_cross_attention=False
)

REGISTRY.register_sequence_model(
    name="bilstm",
    cls=BiLSTMSequenceModel,
    description="Bidirectional LSTM (output dim = d_model)",
    requires_cross_attention=False
)