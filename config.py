from dataclasses import dataclass, field
from typing import List, Optional
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

@dataclass
class CaptchaConfig(HookedTransformerConfig):
    """
    Configuration class for the Captcha architecture.
    
    Inherits from HookedTransformerConfig to ensure compatibility with transformer_lens.
    """
    model_type: str = 'asymmetric-convnext-transformer'
    
    # --- CNN Encoder Parameters (Legacy / Configurable) ---
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [7, 5])
    cnn_strides: List[int] = field(default_factory=lambda: [1, 1])
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32])
    
    # --- Projection Parameters (Legacy) ---
    projection_dims: List[int] = field(default_factory=lambda: [3136, 392, 128, 64])

    # --- Transformer Defaults (Updated for Asymmetric CTC) ---
    n_layers: int = 4           # 4 layers for Encoder
    
    # Updated to match AsymmetricCNNEncoder final stage output (512)
    # If you change this, the model will add a Linear projection layer to bridge dimensions.
    d_model: int = 256          
    
    n_heads: int = 8
    d_head: int = 32            
    d_mlp: int = 1024           
    
    n_ctx: int = 384            
    
    d_vocab: int = 62           # Actual CharSet size (e.g. 0-9, a-z, A-Z). Model adds +1 for CTC Blank.
    
    act_fn: str = "gelu"        
    normalization_type: str = "RMS"
    
    # --- RoPE Configuration ---
    positional_embedding_type: str = "rotary" # Explicitly use RoPE
    rotary_dim: Optional[int] = d_head            # Apply RoPE to the full head dimension (match d_head)
    
    # Standard HookedTransformerConfig defaults
    attention_dir: str = "bidirectional"      # Encoder needs to see whole sequence
    seed: Optional[int] = None

    def __post_init__(self):
        # Calculate derived attributes if they are missing
        if self.d_head is None and self.d_model is not None and self.n_heads is not None:
            self.d_head = self.d_model // self.n_heads
            
        if self.rotary_dim is None and self.positional_embedding_type == "rotary":
            self.rotary_dim = self.d_head

        super().__post_init__()