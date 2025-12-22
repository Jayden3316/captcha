import torch
import torch.nn as nn
from typing import Optional, Any
from jaxtyping import Float

from src.config.config import ModelConfig, TaskType
from src.architecture.registry import REGISTRY

# We need to ensure that the register calls in components are executed.
# Importing them here triggers registration.
import src.architecture.components.encoders
import src.architecture.components.adapters
import src.architecture.components.projectors
import src.architecture.components.sequence_models
import src.architecture.components.heads
from src.architecture.components.projectors import LinearProjector, IdentityProjector
from src.architecture.components.adapters import VerticalFeatureAdapter


class StandardGenerationPipeline(nn.Module):
    """
    Model Pipeline for Generation (Sequence) Tasks.
    PipelineType.STANDARD_GENERATION
    
    Structure:
    1. Encoder (Image -> Spatial Features [B, C, H, W])
    2. Adapter (Spatial Features -> Sequence [B, Seq, Dim])
    3. Projector (Feature Dim -> Model Dim)
    4. Sequence Model (Contextualization)
    5. Head (Model Dim -> Logits [B, Seq, Vocab])
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # --- 1. Encoder ---
        encoder_cls = REGISTRY.get_encoder(config.encoder_type)
        self.encoder = encoder_cls(config.encoder_config)
        
        # --- 2. Adapter ---
        adapter_type = config.adapter_type
        adapter_cls = REGISTRY.get_adapter(adapter_type)
        input_channels = self.encoder.output_channels
        
        # Pass required args to adapter.
        adapter_kwargs = dict(config.adapter_config.__dict__) if config.adapter_config else {}
        
        # Validation for adapters requiring explicit dimensions
        if adapter_type == "vertical_feature":
             if getattr(config.adapter_config, 'output_dim', None) is None:
                 raise ValueError("VerticalFeatureAdapter requires 'output_dim'.")
        
        self.adapter = adapter_cls(input_channels=input_channels, **adapter_kwargs)
             
        # --- 3. Projector ---
        # Adapter output dim -> d_model
        adapter_dim = self.adapter.output_dim
        d_model = config.d_model
        
        if config.projector_type:
             projector_cls = REGISTRY.get_projector(config.projector_type)
             # Projectors expect input_dim and output_dim kwargs
             self.projector = projector_cls(input_dim=adapter_dim, output_dim=d_model) 
        else:
             if adapter_dim != d_model:
                 self.projector = LinearProjector(input_dim=adapter_dim, output_dim=d_model)
             else:
                 self.projector = nn.Identity()
                 
        # --- 4. Sequence Model ---
        if config.sequence_model_type:
            seq_cls = REGISTRY.get_sequence_model(config.sequence_model_type)
            # Ensure d_model is set in sequence config
            if hasattr(config.sequence_model_config, 'd_model'):
                config.sequence_model_config.d_model = d_model
            if hasattr(config.sequence_model_config, 'd_vocab'):
                 config.sequence_model_config.d_vocab = config.d_vocab
                 
            self.sequence_model = seq_cls(config.sequence_model_config)
            
            # Check if sequence model requires cross attention (e.g. Decoder-only DETR style)
            if getattr(self.sequence_model, 'requires_cross_attention', False):
                n_queries = config.sequence_model_config.n_ctx
                self.decoder_queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        else:
            self.sequence_model = nn.Identity()
            
        # --- 5. Head ---
        if config.head_type:
            head_cls = REGISTRY.get_head(config.head_type)
            # Inject generic params into head config if missing
            head_cfg = config.head_config
            if getattr(head_cfg, 'loss_type', None) is None:
                head_cfg.loss_type = config.loss_type
            if getattr(head_cfg, 'd_vocab', None) is None:
                head_cfg.d_vocab = config.d_vocab
            if getattr(head_cfg, 'd_model', None) is None:
                head_cfg.d_model = config.d_model
                
            self.head = head_cls(head_cfg)
        else:
            self.head = nn.Identity()

    def forward(self, image: torch.Tensor, text: Optional[torch.Tensor] = None):
        # 1. Encode
        x = self.encoder(image) # [B, C, H, W]
        
        # 2. Adapt
        x = self.adapter(x) # [B, Seq, Dim]
        
        # 3. Project
        x = self.projector(x) # [B, Seq, d_model]
        
        # 4. Sequence Model
        if isinstance(self.sequence_model, nn.Identity):
            pass
        elif getattr(self.sequence_model, 'requires_cross_attention', False):
            # DETR-style decoder requires queries.
            batch_size = image.shape[0]
            if not hasattr(self, 'decoder_queries'):
                 raise RuntimeError("Sequence model requires cross attention but decoder_queries not initialized.")
                 
            queries = self.decoder_queries.expand(batch_size, -1, -1)
            x = self.sequence_model(queries, encoder_out=x)
        else:
            # Standard encoder-only sequence model
            x = self.sequence_model(x)
            
        # 5. Head
        logits = self.head(x) # [B, Seq, Vocab]
        
        return logits


class StandardClassificationPipeline(nn.Module):
    """
    Model Pipeline for Classification Tasks.
    PipelineType.STANDARD_CLASSIFICATION
    
    Structure:
    1. Encoder (Image -> Spatial Features [B, C, H, W])
    2. Adapter (Spatial Features -> Fixed Vector [B, Dim])
    3. Head (Dim -> Logits [B, NumClasses])
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # --- 1. Encoder ---
        encoder_cls = REGISTRY.get_encoder(config.encoder_type)
        self.encoder = encoder_cls(config.encoder_config)
        
        # --- 2. Adapter ---
        adapter_type = config.adapter_type
        adapter_cls = REGISTRY.get_adapter(adapter_type)
        input_channels = self.encoder.output_channels
        
        # Initialize Adapter
        adapter_kwargs = dict(config.adapter_config.__dict__) if config.adapter_config else {}
        
        # Explicit validation for adapters requiring dimensions
        if adapter_type == "flatten":
             if getattr(config.adapter_config, 'output_dim', None) is None:
                raise ValueError(
                    f"{adapter_type} requires explicit 'output_dim' in adapter_config. "
                    "Please calculate C * H * W for your specific input size and encoder output."
                )
        
        self.adapter = adapter_cls(input_channels=input_channels, **adapter_kwargs)

        input_to_head_dim = self.adapter.output_dim

        # --- 3. Head ---
        if config.head_type:
            head_cls = REGISTRY.get_head(config.head_type)
            head_cfg = config.head_config
            
            # Auto-configure head dimension if possible/needed
            if input_to_head_dim > 0:
                head_cfg.d_model = input_to_head_dim
            
            self.head = head_cls(head_cfg)
        else:
            self.head = nn.Identity()
            
    def forward(self, image: torch.Tensor, text: Optional[torch.Tensor] = None):
        x = self.encoder(image)
        x = self.adapter(x)
        
        # Runtime Validation for FlattenAdapter correctness
        # The Head expects [B, input_to_head_dim]. Adapter produces [B, actual_dim]
        # If there is a mismatch, the linear layer in Head will throw a shape error.
        # We catch it early to provide a helpful message.
        expected_dim = self.adapter.output_dim
        actual_dim = x.shape[1]
        
        if expected_dim != actual_dim:
            raise ValueError(
                f"Dimension Mismatch in Classification Model!\n"
                f"- Configured Adapter Output Dim: {expected_dim}\n"
                f"- Actual Runtime Adapter Output: {actual_dim}\n"
                f"This usually means the 'output_dim' in FlattenAdapterConfig does not match "
                f"the actual Encoder output size (C*H*W).\n"
                f"Encoder Output Shape (before flatten): {self.encoder.output_channels} x H x W (Check spatial dims)"
            )

        logits = self.head(x)
        return logits

class SequenceClassificationPipeline(nn.Module):
    """
    Alternate model pipeline for classification
    PipelineType.SEQUENCE_CLASSIFICATION
    
    - Encoder: [B, C, H, W] -> [B, C, H', W']
    - EncoderAdapter: [B, C, H', W'] -> [B, Seq, Dim]
    - Projector: [B, Seq, Dim] -> [B, Seq, d_model]
    - Sequence Model: [B, Seq, d_model] -> [B, Seq, d_model]
    - SequenceModelAdapter: [B, Seq, d_model] -> [B, Dim]
    - Head: [B, Dim] -> [B, NumClasses]
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # --- 1. Encoder ---
        encoder_cls = REGISTRY.get_encoder(config.encoder_type)
        self.encoder = encoder_cls(config.encoder_config)

        # --- 2. EncoderAdapter ---
        encoder_adapter_type = config.encoder_adapter_type 
        encoder_adapter_cls = REGISTRY.get_adapter(encoder_adapter_type) # Reusing adapter registry
        input_channels = self.encoder.output_channels

        # Initialize EncoderAdapter
        encoder_adapter_kwargs = dict(config.encoder_adapter_config.__dict__) if config.encoder_adapter_config else {}

        if encoder_adapter_type == "vertical_feature":
            if getattr(config.encoder_adapter_config, 'output_dim', None) is None:
                raise ValueError(f"{encoder_adapter_type} requires explicit 'output_dim' in encoder_adapter_config.")
        
        self.encoder_adapter = encoder_adapter_cls(input_channels=input_channels, **encoder_adapter_kwargs)

        input_to_head_dim = self.encoder_adapter.output_dim

        input_to_head_dim = self.encoder_adapter.output_dim

        # --- 3. Projector ---
        # Adapter output dim -> d_model
        adapter_dim = self.encoder_adapter.output_dim
        d_model = config.d_model
        
        if config.projector_type:
             projector_cls = REGISTRY.get_projector(config.projector_type)
             self.projector = projector_cls(input_dim=adapter_dim, output_dim=d_model) 
        else:
             if adapter_dim != d_model:
                 self.projector = LinearProjector(input_dim=adapter_dim, output_dim=d_model)
             else:
                 self.projector = nn.Identity()

        # --- 4. Sequence Model ---
        if config.sequence_model_type:
            seq_cls = REGISTRY.get_sequence_model(config.sequence_model_type)
            # Ensure d_model sync
            if hasattr(config.sequence_model_config, 'd_model'):
                config.sequence_model_config.d_model = d_model
                 
            self.sequence_model = seq_cls(config.sequence_model_config)
        else:
            self.sequence_model = nn.Identity()
            
        # --- 5. Sequence Adapter (Sequence -> Vector) ---
        # Used to pool sequence output to single vector for classification
        sequence_adapter_type = config.sequence_adapter_type
        if sequence_adapter_type:
            seq_adapter_cls = REGISTRY.get_adapter(sequence_adapter_type)
            # Sequence Pooling Adapter needs input_channels (dim)
            seq_adapter_kwargs = dict(config.sequence_adapter_config.__dict__) if config.sequence_adapter_config else {}
            self.sequence_adapter = seq_adapter_cls(input_channels=d_model, **seq_adapter_kwargs)
            input_to_head_dim = self.sequence_adapter.output_dim
        else:
            # Fallback: if no sequence adapter, assume sequence model returns valid dim or we just take mean?
            # Or if seq model is identity, we still have [B, Seq, Dim]. We must pool.
            # Default to Mean Pooling if undefined but task is classification?
            # Or raise error if strict?
            # Let's default to a simple mean pool if missing config but enforce explicit config usually.
            from src.architecture.components.adapters import SequencePoolingAdapter
            self.sequence_adapter = SequencePoolingAdapter(input_channels=d_model, pool_type="mean")
            input_to_head_dim = d_model

        # --- 6. Head ---
        if config.head_type:
            head_cls = REGISTRY.get_head(config.head_type)
            head_cfg = config.head_config
            
            # Auto-configure head dimension
            if input_to_head_dim > 0:
                head_cfg.d_model = input_to_head_dim
            
            self.head = head_cls(head_cfg)
        else:
            self.head = nn.Identity()

    def forward(self, image: torch.Tensor, text: Optional[torch.Tensor] = None):
        # 1. Encode [B, C, H, W]
        x = self.encoder(image) 
        
        # 2. Encoder Adapter [B, Seq, Dim]
        x = self.encoder_adapter(x) 
        
        # 3. Projector [B, Seq, d_model]
        x = self.projector(x)
        
        # 4. Sequence Model [B, Seq, d_model]
        x = self.sequence_model(x)
        
        # 5. Sequence Adapter [B, d_model] (Pooling)
        x = self.sequence_adapter(x)
        
        # 6. Head [B, NumClasses]
        logits = self.head(x)
        
        return logits
def CaptchaModel(config: ModelConfig) -> nn.Module:
    """
    Factory function to create the appropriate CaptchaModel based on PipelineType.
    """
    from src.config.config import PipelineType
    
    if config.pipeline_type == PipelineType.STANDARD_GENERATION:
        return StandardGenerationPipeline(config)
    elif config.pipeline_type == PipelineType.STANDARD_CLASSIFICATION:
        return StandardClassificationPipeline(config)
    elif config.pipeline_type == PipelineType.SEQUENCE_CLASSIFICATION:
        return SequenceClassificationPipeline(config)
    else:
        # Fallback based on task type (deprecated path, but keeping for safety)
        if config.task_type == TaskType.GENERATION:
             return StandardGenerationPipeline(config)
        elif config.task_type == TaskType.CLASSIFICATION:
             return StandardClassificationPipeline(config)
             
    raise ValueError(f"Unknown pipeline type: {config.pipeline_type} and task type: {config.task_type}")
        
