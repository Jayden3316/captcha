"""
Projector implementations for bridging encoder and transformer dimensions.
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from .base import BaseProjector
from ..registry import REGISTRY

Tensor = torch.Tensor


class LinearProjector(BaseProjector):
    """
    Simple linear projection.
    
    Efficient single-layer mapping from encoder dim to d_model.
    Use when encoder dim and d_model are reasonably close.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: Float[Tensor, "batch seq input_dim"]) -> Float[Tensor, "batch seq output_dim"]:
        return self.linear(x)


class MLPProjector(BaseProjector):
    """
    Multi-layer projector with non-linearity.
    
    Provides more capacity for bridging large dimensional gaps.
    Use when encoder dim and d_model are very different.
    
    Args:
        input_dim: Encoder output dimension
        output_dim: Target d_model dimension
        hidden_dim: Hidden layer dimension (default: average of input and output)
        num_layers: Number of layers (default: 2)
        activation: Activation function (default: GELU)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        activation: str = 'gelu'
    ):
        super().__init__(input_dim, output_dim)
        
        # Default hidden dim is average of input and output
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        # Build layer dimensions
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        # Get activation function
        act_fn = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh
        }.get(activation.lower(), nn.GELU)
        
        # Build MLP
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # No activation on final layer
            if i < len(dims) - 2:
                layers.append(act_fn())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: Float[Tensor, "batch seq input_dim"]) -> Float[Tensor, "batch seq output_dim"]:
        return self.mlp(x)


class IdentityProjector(BaseProjector):
    """
    No-op projector for when dimensions already match.
    
    Use when encoder output_dim == d_model to save parameters.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        if input_dim != output_dim:
            raise ValueError(
                f"IdentityProjector requires matching dimensions, "
                f"got input_dim={input_dim}, output_dim={output_dim}"
            )
    
    def forward(self, x: Float[Tensor, "batch seq input_dim"]) -> Float[Tensor, "batch seq output_dim"]:
        return x


class BottleneckProjector(BaseProjector):
    """
    Projector with bottleneck architecture.
    
    Projects through a narrow bottleneck before expanding to target dimension.
    Can help with regularization and feature compression.
    
    Args:
        input_dim: Encoder output dimension
        output_dim: Target d_model dimension
        bottleneck_dim: Bottleneck dimension (default: min(input, output) / 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int = None
    ):
        super().__init__(input_dim, output_dim)
        
        if bottleneck_dim is None:
            bottleneck_dim = min(input_dim, output_dim) // 2
        
        self.compress = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.expand = nn.Linear(bottleneck_dim, output_dim)
    
    def forward(self, x: Float[Tensor, "batch seq input_dim"]) -> Float[Tensor, "batch seq output_dim"]:
        x = self.compress(x)
        x = self.act(x)
        x = self.expand(x)
        return x


class ResidualProjector(BaseProjector):
    """
    Projector with residual connection.
    
    Uses skip connection to preserve information.
    Only works when input_dim <= output_dim.
    
    Args:
        input_dim: Encoder output dimension
        output_dim: Target d_model dimension
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None
    ):
        super().__init__(input_dim, output_dim)
        
        if input_dim > output_dim:
            raise ValueError(
                "ResidualProjector requires input_dim <= output_dim for skip connection"
            )
        
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        # Main path
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Skip connection (pad if dimensions don't match)
        self.skip = nn.Identity()
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: Float[Tensor, "batch seq input_dim"]) -> Float[Tensor, "batch seq output_dim"]:
        identity = self.skip(x)
        
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        
        return out + identity


# ========== REGISTRATION ==========

REGISTRY.register_projector(
    name='linear',
    cls=LinearProjector,
    description='Simple linear projection'
)

REGISTRY.register_projector(
    name='mlp',
    cls=MLPProjector,
    description='Multi-layer projector with non-linearity',
    supports_kwargs=['hidden_dim', 'num_layers', 'activation']
)

REGISTRY.register_projector(
    name='identity',
    cls=IdentityProjector,
    description='No-op projector (requires matching dimensions)'
)

REGISTRY.register_projector(
    name='bottleneck',
    cls=BottleneckProjector,
    description='Projector with compression bottleneck',
    supports_kwargs=['bottleneck_dim']
)

REGISTRY.register_projector(
    name='residual',
    cls=ResidualProjector,
    description='Projector with residual connection',
    supports_kwargs=['hidden_dim'],
    constraint='Requires input_dim <= output_dim'
)