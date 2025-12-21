"""
Output head implementations with registration.
All heads inherit from BaseHead.
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from typing import Dict, Any, Optional

from .base import BaseHead
from ..registry import REGISTRY
from ...config import LinearHeadConfig, MLPHeadConfig, ClassificationHeadConfig

Tensor = torch.Tensor

class LinearHead(BaseHead):
    """
    Simple linear projection head.
    Supports both standard decoding (d_model -> d_vocab) 
    and CTC decoding (d_model -> d_vocab + 1).
    """
    def __init__(self, cfg: LinearHeadConfig):
        super().__init__()
        self.cfg = cfg
        
        # Determine output dimension based on head/loss type
        self.is_ctc = cfg.head_type == 'ctc' or cfg.loss_type == 'ctc'
        self.vocab_size = cfg.d_vocab
        self.out_dim = self.vocab_size + 1 if self.is_ctc else self.vocab_size
        
        self.projector = nn.Linear(cfg.d_model, self.out_dim)

    @property
    def decoding_type(self) -> str:
        return 'ctc' if self.is_ctc else 'autoregressive'

    @property
    def output_shape_info(self) -> Dict[str, Any]:
        return {
            'type': 'sequence',
            'size': self.out_dim,
            'includes_blank': self.is_ctc
        }

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq vocab_size"]:
        return self.projector(x)


class MLPHead(BaseHead):
    """
    MLP-based sequence head.
    Applies an MLP per token before projecting to vocab.
    Useful when the sequence model output needs non-linear processing before classification.
    """
    def __init__(self, cfg: MLPHeadConfig):
        super().__init__()
        self.cfg = cfg
        
        self.is_ctc = cfg.head_type == 'ctc' or cfg.loss_type == 'ctc'
        self.vocab_size = cfg.d_vocab
        self.out_dim = self.vocab_size + 1 if self.is_ctc else self.vocab_size
        
        # Configurable MLP parameters with defaults
        # We look for 'head_num_layers' etc in config, or fall back to defaults
        num_layers = getattr(cfg, 'head_num_layers', 2)
        hidden_dim = getattr(cfg, 'head_hidden_dim', cfg.d_model * 2)
        dropout = getattr(cfg, 'head_dropout', 0.1)
        
        layers = []
        input_dim = cfg.d_model
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        # Final projection
        layers.append(nn.Linear(input_dim, self.out_dim))
        
        self.mlp = nn.Sequential(*layers)

    @property
    def decoding_type(self) -> str:
        return 'ctc' if self.is_ctc else 'autoregressive'

    @property
    def output_shape_info(self) -> Dict[str, Any]:
        return {
            'type': 'sequence',
            'size': self.out_dim,
            'includes_blank': self.is_ctc
        }

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq vocab_size"]:
        return self.mlp(x)


class ClassificationHead(BaseHead):
    """
    Global classification head.
    Pools the sequence (Mean/Max) and projects to num_classes.
    """
    def __init__(self, cfg: ClassificationHeadConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.num_classes is None:
            raise ValueError("ClassificationHead requires 'num_classes' to be set in config")
            
        self.num_classes = cfg.num_classes
        self.pooling_type = getattr(cfg, 'pooling_type', 'mean')
        
        hidden_dim = getattr(cfg, 'head_hidden_dim', cfg.d_model)
        
        # Simple MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes)
        )

    @property
    def decoding_type(self) -> str:
        return 'classification'

    @property
    def output_shape_info(self) -> Dict[str, Any]:
        return {
            'type': 'single',
            'size': self.num_classes,
            'includes_blank': False
        }

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch num_classes"]:
        # x: [Batch, Seq, Dim]
        
        if self.pooling_type == 'mean':
            # Masking could be applied here if padding mask is available
            pooled = x.mean(dim=1)
        elif self.pooling_type == 'max':
            pooled = x.max(dim=1)[0]
        elif self.pooling_type == 'first':
            pooled = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
        return self.mlp(pooled)


# ============================================================================
# REGISTRATION
# ============================================================================

REGISTRY.register_head(
    name="linear",
    cls=LinearHead,
    description="Simple linear projection (d_model -> vocab)",
    decoding_type="autoregressive/ctc",
    compatible_losses=["ctc", "cross_entropy"]
)

REGISTRY.register_head(
    name="ctc",
    cls=LinearHead, # Reuses LinearHead but conceptually distinct in config
    description="Linear projection for CTC (d_model -> vocab + 1)",
    decoding_type="ctc",
    compatible_losses=["ctc"]
)

REGISTRY.register_head(
    name="mlp",
    cls=MLPHead,
    description="MLP sequence head (d_model -> hidden -> vocab)",
    decoding_type="autoregressive/ctc",
    compatible_losses=["ctc", "cross_entropy"]
)

REGISTRY.register_head(
    name="classification",
    cls=ClassificationHead,
    description="Global classification head with pooling",
    decoding_type="classification",
    compatible_losses=["cross_entropy", "focal"]
)