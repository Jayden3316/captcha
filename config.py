from dataclasses import dataclass, field
from typing import List
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

@dataclass
class CaptchaConfig(HookedTransformerConfig):
    """
    Configuration class for the Captcha architecture described in README.md.
    
    Inherits from HookedTransformerConfig to ensure compatibility with transformer_lens.
    """
    # --- CNN Encoder Parameters ---
    # Layer 1: Conv2D [7, 1, 16], MaxPool stride 2
    # Layer 2: Conv2D [5, 1, 32], MaxPool stride 2
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [7, 5])
    cnn_strides: List[int] = field(default_factory=lambda: [1, 1])
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32])
    
    # --- Projection Parameters ---
    # Maps flattened CNN features to Transformer d_model
    # Input: 3136 (14 * 7 * 32) -> 392 -> 128 -> 64
    projection_dims: List[int] = field(default_factory=lambda: [3136, 392, 128, 64])

    # --- Transformer Defaults (overriding HookedTransformerConfig) ---
    n_layers: int = 4           # 4 layers for Encoder, 4 for Decoder
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16            # d_model / n_heads = 64 / 4 = 16
    d_mlp: int = 256            # 4 * d_model = 256
    
    n_ctx: int = 128             # Corresponds to seq_len in README (longest word + buffer)
    d_vocab: int = 62           # Placeholder: letters + <PAD>. Update based on actual vocab size.
    
    act_fn: str = "gelu"        # README specifies GeGELU; gelu is the standard approximation
    normalization_type: str = "RMS"
    positional_embedding_type: str = "rotary"
    
    # Standard HookedTransformerConfig defaults that match our needs
    # attention_dir="causal" (default) is correct for the Decoder
    # seed=None (default)

    def __post_init__(self):
        super().__post_init__()