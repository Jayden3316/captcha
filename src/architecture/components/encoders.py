"""
Encoder implementations with clean interfaces and registration.
All encoders inherit from BaseEncoder and register themselves.
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from .base import BaseEncoder
from ..registry import REGISTRY

Tensor = torch.Tensor


# ========== UTILITY LAYERS ==========

class DropPath(nn.Module):
    """Stochastic depth for residual blocks."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for (N, C, H, W) format."""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNextBlock(nn.Module):
    """ConvNeXt block with depthwise and pointwise convolutions."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class AsymmetricDownsample(nn.Module):
    """
    Downsamples height by 2, keeps width unchanged.
    Preserves sequence length for CTC while reducing vertical dimension.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = LayerNorm2d(dim_in, eps=1e-6)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        return self.conv(self.norm(x))


# ========== ENCODER IMPLEMENTATIONS ==========

class AsymmetricConvNextEncoder(BaseEncoder):
    """
    ConvNeXt-style encoder with asymmetric downsampling.
    
    Preserves horizontal resolution for sequence modeling while
    aggressively reducing vertical dimension.
    
    Architecture:
        Input: [B, 3, 80, W]
        Stem (4x4 stride 4): [B, 64, 20, W/4]
        Stage 1 (2 blocks): [B, 64, 20, W/4]
        Downsample H: [B, 128, 10, W/4]
        Stage 2 (2 blocks): [B, 128, 10, W/4]
        Downsample H: [B, 256, 5, W/4]
        Stage 3 (6 blocks): [B, 256, 5, W/4]
        Downsample H: [B, 512, 2, W/4]
        Stage 4 (2 blocks): [B, 512, 2, W/4]
        Mean pool H: [B, 512, W/4]
        Transpose: [B, W/4, 512]
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dims = [64, 128, 256, 512]
        
        # Stem: 4x4 patches with stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.stage1 = nn.Sequential(*[ConvNextBlock(dims[0]) for _ in range(2)])
        
        # Asymmetric downsampling stages
        self.down2 = AsymmetricDownsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvNextBlock(dims[1]) for _ in range(2)])
        
        self.down3 = AsymmetricDownsample(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[ConvNextBlock(dims[2]) for _ in range(6)])
        
        self.down4 = AsymmetricDownsample(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[ConvNextBlock(dims[3]) for _ in range(2)])
        
        self._output_dim = dims[3]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Float[Tensor, "batch 3 80 width"]) -> Float[Tensor, "batch seq 512"]:
        # Encoding stages
        x = self.stem(x)        # [B, 64, 20, W/4]
        x = self.stage1(x)
        
        x = self.down2(x)       # [B, 128, 10, W/4]
        x = self.stage2(x)
        
        x = self.down3(x)       # [B, 256, 5, W/4]
        x = self.stage3(x)
        
        x = self.down4(x)       # [B, 512, 2, W/4]
        x = self.stage4(x)
        
        # Collapse vertical dimension
        x = x.mean(dim=2)       # [B, 512, W/4]
        
        # Transpose to sequence format
        x = x.permute(0, 2, 1)  # [B, W/4, 512]
        
        return x
    
    def get_sequence_length(self, image_width: int) -> int:
        """Sequence length after stem stride of 4."""
        return image_width // 4


class LegacyCNNEncoder(BaseEncoder):
    """
    Legacy CNN encoder with unfold-based patch extraction.
    
    Uses traditional conv-pool structure and extracts 14x7 patches.
    Width must be 28k + 14 for clean patch extraction.
    
    Architecture:
        Input: [B, 3, 70, W]
        Conv1 (7x7): [B, 16, 64, W-6]
        Pool (2x2): [B, 16, 32, (W-6)/2]
        Conv2 (5x5): [B, 32, 28, ...]
        Pool (2x2): [B, 32, 14, W_final]
        Unfold patches (7 wide): [B, num_patches, 32*14*7=3136]
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.act = nn.SiLU()
        
        self._output_dim = 3136  # 32 * 14 * 7

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Float[Tensor, "batch 3 70 width"]) -> Float[Tensor, "batch seq 3136"]:
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        
        # Extract 14x7 patches from [B, 32, 14, W_final]
        patches = x.unfold(3, 7, 7)  # [B, 32, 14, num_patches, 7]
        
        # Rearrange to [B, num_patches, 3136]
        patches = patches.permute(0, 3, 1, 2, 4).contiguous()
        b, num_patches, c, h, w = patches.shape
        out = patches.view(b, num_patches, c * h * w)
        
        return out
    
    def get_sequence_length(self, image_width: int) -> int:
        """Calculate sequence length from width formula."""
        # After conv1 and pool1: (width - 6) / 2
        # After conv2 and pool2: further reductions
        # Then unfold with stride 7
        # This is approximate - exact formula depends on all convolution parameters
        return ((image_width - 6) // 2 - 4) // 2 // 7


# ========== REGISTRATION ==========

REGISTRY.register_encoder(
    name='asymmetric_convnext',
    cls=AsymmetricConvNextEncoder,
    description='ConvNeXt with asymmetric downsampling for CTC',
    compatible_heights=[80],
    width_divisor=4,
    sequence_formula='W/4'
)

REGISTRY.register_encoder(
    name='legacy_cnn',
    cls=LegacyCNNEncoder,
    description='Legacy CNN with 14x7 patch extraction',
    compatible_heights=[70],
    width_divisor=28,
    width_bias=14,
    sequence_formula='(((W-6)/2-4)/2)/7',
    note='Width should be 28k + 14'
)