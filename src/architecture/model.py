import torch
import torch.nn as nn
from typing import Optional, Any
from jaxtyping import Float

from captcha_ocr.config.config import ModelConfig
from captcha_ocr.architecture.registry import REGISTRY

# We need to ensure that the register calls in components are executed.
# Importing them here triggers registration.
import captcha_ocr.architecture.components.encoders
import captcha_ocr.architecture.components.projectors
import captcha_ocr.architecture.components.sequence_models
import captcha_ocr.architecture.components.heads
from captcha_ocr.architecture.components.projectors import LinearProjector

class CaptchaModel(nn.Module):
    """
    Clean Captcha Model Pipeline using the Component Registry.
    
    Structure:
    1. Encoder (Image -> Features)
    2. Projector (Features -> Model Dim)
    3. Sequence Model (Contextualization)
    4. Head (Model Dim -> Logits/Classes)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # --- 1. Encoder ---
        encoder_cls = REGISTRY.get_encoder(config.encoder_type)
        self.encoder = encoder_cls(config.encoder_config)
        
        # --- 2. Projector ---
        encoder_dim = self.encoder.output_dim
        d_model = config.d_model
        
        if config.projector_type:
             projector_cls = REGISTRY.get_projector(config.projector_type)
             # Projectors expect input_dim and output_dim kwargs
             self.projector = projector_cls(input_dim=encoder_dim, output_dim=d_model) 
        else:
            # Default logic
            if encoder_dim != d_model:
                self.projector = LinearProjector(input_dim=encoder_dim, output_dim=d_model)
            else:
                self.projector = nn.Identity()
                
        # --- 3. Sequence Model ---
        if config.sequence_model_type:
            seq_cls = REGISTRY.get_sequence_model(config.sequence_model_type)
            # Ensure d_model is set in sequence config
            if hasattr(config.sequence_model_config, 'd_model'):
                config.sequence_model_config.d_model = d_model
            # Ensure d_vocab is set if needed (TransformerLens config might use it)
            if hasattr(config.sequence_model_config, 'd_vocab'):
                 config.sequence_model_config.d_vocab = config.d_vocab
                 
            self.sequence_model = seq_cls(config.sequence_model_config)
            
            # Check if sequence model requires cross attention (e.g. Decoder-only DETR style)
            if getattr(self.sequence_model, 'requires_cross_attention', False):
                # Initialize learned queries for the decoder
                # n_ctx here acts as the fixed number of query slots (e.g. max_length)
                n_queries = config.sequence_model_config.n_ctx
                self.decoder_queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        else:
            self.sequence_model = nn.Identity()
            
        # --- 4. Head ---
        if config.head_type:
            head_cls = REGISTRY.get_head(config.head_type)
            # Inject generic params into head config if missing or None
            if getattr(config.head_config, 'loss_type', None) is None:
                config.head_config.loss_type = config.loss_type
            if getattr(config.head_config, 'd_vocab', None) is None:
                config.head_config.d_vocab = config.d_vocab
            if getattr(config.head_config, 'd_model', None) is None:
                config.head_config.d_model = config.d_model
                
            self.head = head_cls(config.head_config)
        else:
            self.head = nn.Identity()

    def forward(self, image: torch.Tensor, text: Optional[torch.Tensor] = None):
        # 1. Encode
        x = self.encoder(image) 
        
        # 2. Project
        x = self.projector(x)
        
        # 3. Sequence Model
        # Check if sequence model requires cross attention
        if isinstance(self.sequence_model, nn.Identity):
            pass
        elif getattr(self.sequence_model, 'requires_cross_attention', False):
            # DETR-style decoder requires queries.
            # We use the learned queries initialized in __init__
            batch_size = image.shape[0]
            if not hasattr(self, 'decoder_queries'):
                 raise RuntimeError("Sequence model requires cross attention but decoder_queries not initialized.")
                 
            queries = self.decoder_queries.expand(batch_size, -1, -1)
            x = self.sequence_model(queries, encoder_out=x)
        else:
            # Standard encoder-only sequence model
            x = self.sequence_model(x)
            
        # 4. Head
        logits = self.head(x)
        
        return logits
