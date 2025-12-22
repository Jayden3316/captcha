"""
Adapter implementations for converting spatial features to task-specific formats.
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from .base import BaseGenerationAdapter, BaseClassificationAdapter
from ..registry import REGISTRY

Tensor = torch.Tensor

# ========== GENERATION ADAPTERS ==========

class VerticalFeatureAdapter(BaseGenerationAdapter):
    """
    Produces features by flattening in the following manner:
    [B, C, H, W] -> [B, W//factor, C * H * factor]
    
    Combines contiguous receptive fields into a single embedding.
    Calculates `factor` dynamically based on provided `output_dim` and input tensor shape.
    
    D_model (output_dim) must be: C * H * factor.
    """
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__(input_channels)
        self._output_dim = output_dim
        self.factor = None
        
    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch seq dim"]:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Calculate factor based on expected output_dim
        # output_dim = C * H * factor
        # factor = output_dim / (C * H)
        feature_size_per_step = C * H
        
        if self._output_dim % feature_size_per_step != 0:
            raise ValueError(f"Output dim {self._output_dim} must be divisible by C*H ({C}*{H}={feature_size_per_step})")
            
        self.factor = self._output_dim // feature_size_per_step
        
        if W % self.factor != 0:
             # Handle padding if needed, or raise error. 
             # For now, simplistic view/reshape requires divisibility.
             raise ValueError(f"Input width {W} must be divisible by calculated factor {self.factor}")

        # [B, C, H, W] -> [B, C, H, W//F, F]
        x = x.view(B, C, H, W // self.factor, self.factor)
        # Permute to [B, W//F, C, H, F] -> [B, W//F, C*H*F]
        x = x.permute(0, 3, 1, 2, 4).flatten(2)
        return x

    def get_sequence_length(self, input_width: int) -> int:
        if self.factor is None:
             # We haven't run forward yet, so we don't know H or factor.
             # This is a limitation. 
             # For planning/debug, we might return input_width (factor=1).
             # Or we should have required factor in __init__.
             return input_width
        return input_width // self.factor


# ========== CLASSIFICATION ADAPTERS ==========

class FlattenAdapter(BaseClassificationAdapter):
    """
    Flattens all spatial dimensions.
    
    [B, C, H, W] -> [B, C*H*W]
    """
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__(input_channels)
        self._output_dim = output_dim
        
    @property
    def output_dim(self) -> int:
        return self._output_dim
        
    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch dim"]:
        return x.flatten(1)


class GlobalPoolingAdapter(BaseClassificationAdapter):
    """
    Global Average or Max Pooling.
    
    [B, C, H, W] -> [B, C]
    """
    def __init__(self, input_channels: int, pool_type: str = "avg"):
        super().__init__(input_channels)
        self.pool_type = pool_type
        
    @property
    def output_dim(self) -> int:
        return self.input_channels
        
    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch dim"]:
        if self.pool_type == "avg":
            return x.mean(dim=(2, 3))
        elif self.pool_type == "max":
            return x.amax(dim=(2, 3))
        else:
             raise ValueError(f"Unknown pool type: {self.pool_type}")


# ========== REGISTRATION ==========
# We'll do registration in registry.py or here if we import REGISTRY.
# But registry.py imports THIS file usually (or we avoid circular imports by registering in registry.py or main).
# To follow pattern in encoders.py, we register here.
# But wait, registry.py imports base.py. If encoders.py imports registry, that's fine.
# Let's check imports in encoders.py: `from .base import ...` and `from ..registry import REGISTRY`.
# So we can do the same.

REGISTRY.register_adapter(
    name="vertical_feature",
    cls=VerticalFeatureAdapter,
    description="Flattens vertical dim and groups h-receptive fields.",
    type="generation"
)

REGISTRY.register_adapter(
    name="flatten",
    cls=FlattenAdapter,
    description="Flattens all spatial dimensions",
    type="classification"
)

REGISTRY.register_adapter(
    name="global_pool",
    cls=GlobalPoolingAdapter,
    description="Global pooling (avg or max)",
    type="classification"
)

class SequencePoolingAdapter(BaseClassificationAdapter):
    """
    Pools a sequence to a single vector.
    [B, Seq, Dim] -> [B, Dim]
    """
    def __init__(self, input_channels: int, pool_type: str = "mean"):
        # Note: input_channels here refers to d_model (Dim)
        super().__init__(input_channels)
        self.pool_type = pool_type
        
    @property
    def output_dim(self) -> int:
        return self.input_channels
        
    def forward(self, x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch dim"]:
        if self.pool_type == "mean":
            return x.mean(dim=1)
        elif self.pool_type == "max":
            return x.amax(dim=1)
        elif self.pool_type == "last":
            return x[:, -1, :]
        elif self.pool_type == "first":
            return x[:, 0, :]
        else:
             raise ValueError(f"Unknown pool type: {self.pool_type}")

REGISTRY.register_adapter(
    name="sequence_pool",
    cls=SequencePoolingAdapter,
    description="Pools sequence to vector (mean, max, first, last)",
    type="classification"
)
