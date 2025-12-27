"""
Encoder implementations with clean interfaces and registration.
All encoders inherit from BaseImageEncoder and register themselves.
"""
import torch
import torch.nn as nn
from jaxtyping import Float

from .base import BaseImageEncoder
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

class ConvNextSeparated(nn.Module):
    """
    ConvNextSeparated is just like the ConvNextBlock
    It just adds the convolution output to the residual stream before the MLP.
    It is typically used with a lower expansion ratio (say 1)
    Layer norm is applied before the convolution as well as before the MLP
    """
    def __init__(self, dim, drop_path=0, layer__scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x + input
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x
    
class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        self.norm = LayerNorm2d(dim_out, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = input + x
        return x

class DownsampleBlock(nn.Module):
    """
    Generic downsampling block with configurable stride/kernel.
    """
    def __init__(self, dim_in, dim_out, stride, kernel_size=None, padding=(0, 0)):
        super().__init__()
        self.norm = LayerNorm2d(dim_in, eps=1e-6)
        # Default kernel_size to stride if not specified
        if kernel_size is None:
            kernel_size = stride
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(self.norm(x))


# ========== ENCODER IMPLEMENTATIONS ==========
# TODO: reimplement these with an abstract class
# will help keep the code DRY
# and allow for swapping out convolution blocks in config

class ConvNextEncoder(BaseImageEncoder):
    """
    ConvNeXt-style encoder with configurable downsampling.
    
    Architecture:
        Input: [B, 3, 80, W]
        ...
        Output: [B, 512, 2, W/k]
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dims = cfg.dims
        counts = cfg.stage_block_counts

        strides = getattr(cfg, 'downsample_strides', [(2, 2), (2, 2), (2, 2)])
        kernels = getattr(cfg, 'downsample_kernels', None)
        padding = getattr(cfg, 'downsample_padding', [(0, 0), (0, 0), (0, 0)])
        if kernels is None:
            kernels = [None] * 3
        
        if len(strides) != 3:
             raise ValueError(f"Expected 3 downsample strides, got {len(strides)}")
        if len(kernels) != 3:
             raise ValueError(f"Expected 3 downsample kernels, got {len(kernels)}")
        if len(kernels) != 3:
             raise ValueError(f"Expected 3 downsample padding, got {len(padding)}")

        # Stem: 4x4 patches with stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=cfg.stem_kernel_size, stride=cfg.stem_stride, padding=cfg.stem_padding),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.stage1 = nn.Sequential(*[ConvNextBlock(dims[0], 
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[0])])
        
        self.down2 = DownsampleBlock(dims[0], dims[1], stride=strides[0], kernel_size=kernels[0], padding=padding[0])
        self.stage2 = nn.Sequential(*[ConvNextBlock(dims[1],
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[1])])
        
        self.down3 = DownsampleBlock(dims[1], dims[2], stride=strides[1], kernel_size=kernels[1], padding=padding[1])
        self.stage3 = nn.Sequential(*[ConvNextBlock(dims[2],
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[2])])
        

        self.down4 = DownsampleBlock(dims[2], dims[3], stride=strides[2], kernel_size=kernels[2], padding=padding[2])
        self.stage4 = nn.Sequential(*[ConvNextBlock(dims[3],
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[3])])
        
        self._output_channels = dims[3]
        
        # Calculate width reduction factor
        # Stem stride (4) * downsample strides X components
        self.width_downsample_factor = cfg.stem_stride
        for s in strides:
            ts = s if isinstance(s, (tuple, list)) else (s, s)
            self.width_downsample_factor *= ts[1]

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def forward(self, x: Float[Tensor, "batch 3 80 width"]) -> Float[Tensor, "batch 512 2 width_sub"]:
        # Encoding stages
        x = self.stem(x)        # [B, 64, H/4, W/4]
        x = self.stage1(x)
        
        x = self.down2(x)
        x = self.stage2(x)
        
        x = self.down3(x)
        x = self.stage3(x)
        
        x = self.down4(x)
        x = self.stage4(x)
        
        return x
    
    def get_downsample_factor(self) -> int:
        """Returns the total width downsample factor."""
        return self.width_downsample_factor


class ResNetEncoder(BaseImageEncoder):
    """
    A ResNet type encoder with configurable downsampling.

    Architecture:
        Input: [B, 3, 64, 192]
        ...
        Output: [B, 512, 4, 12] (assuming H=64, W=192)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dims = cfg.dims
        counts = cfg.stage_block_counts
        
        # Default to symmetric (2,2) if not specified
        strides = getattr(cfg, 'downsample_strides', [(2, 2), (2, 2), (1, 1)])
        kernels = getattr(cfg, 'downsample_kernels', None)
        padding = getattr(cfg, 'downsample_padding', [(0, 0), (0, 0), (0, 0)])
        if kernels is None:
            kernels = [None] * 3

        if len(strides) != 3:
             raise ValueError(f"Expected 3 downsample strides, got {len(strides)}")
        if len(kernels) != 3:
             raise ValueError(f"Expected 3 downsample kernels, got {len(kernels)}")
        if len(padding) != 3:
             raise ValueError(f"Expected 3 downsample padding, got {len(padding)}")

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.stem_in_channels, dims[0], kernel_size=cfg.stem_kernel_size, stride=cfg.stem_stride, padding=cfg.stem_padding),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.stage1 = nn.Sequential(*[ResNetBlock(dims[0], dims[0]) for _ in range(counts[0])])
        self.down2 = DownsampleBlock(dims[0], dims[1], stride=strides[0], kernel_size=kernels[0], padding=padding[0])
        self.stage2 = nn.Sequential(*[ResNetBlock(dims[1], dims[1]) for _ in range(counts[1])])
        self.down3 = DownsampleBlock(dims[1], dims[2], stride=strides[1], kernel_size=kernels[1], padding=padding[1])
        self.stage3 = nn.Sequential(*[ResNetBlock(dims[2], dims[2]) for _ in range(counts[2])])
        self.down4 = DownsampleBlock(dims[2], dims[3], stride=strides[2], kernel_size=kernels[2], padding=padding[2])
        self.stage4 = nn.Sequential(*[ResNetBlock(dims[3], dims[3]) for _ in range(counts[3])])
        self._output_channels = dims[3]

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch channel_out height_out width_out"]:
        #print(f"DEBUG: ResNet Input: {x.shape}")
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.down4(x)
        x = self.stage4(x)
        #print(f"DEBUG: ResNet Output: {x.shape}")
        return x

class ConvNextSeparatedEncoder(BaseImageEncoder):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dims = cfg.dims
        counts = cfg.stage_block_counts

        strides = getattr(cfg, 'downsample_strides', [(2, 2), (2, 2), (2, 2)])
        kernels = getattr(cfg, 'downsample_kernels', None)
        padding = getattr(cfg, 'downsample_padding', [(0, 0), (0, 0), (0, 0)])
        if kernels is None:
            kernels = [None] * 3
        
        if len(strides) != 3:
             raise ValueError(f"Expected 3 downsample strides, got {len(strides)}")
        if len(kernels) != 3:
             raise ValueError(f"Expected 3 downsample kernels, got {len(kernels)}")
        if len(kernels) != 3:
             raise ValueError(f"Expected 3 downsample padding, got {len(padding)}")

        # Stem: 4x4 patches with stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=cfg.stem_kernel_size, stride=cfg.stem_stride, padding=cfg.stem_padding),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.stage1 = nn.Sequential(*[ConvNextBlock(dims[0], 
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[0])])
        
        self.down2 = DownsampleBlock(dims[0], dims[1], stride=strides[0], kernel_size=kernels[0], padding=padding[0])
        self.stage2 = nn.Sequential(*[ConvNextBlock(dims[1],
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[1])])
        
        self.down3 = DownsampleBlock(dims[1], dims[2], stride=strides[1], kernel_size=kernels[1], padding=padding[1])
        self.stage3 = nn.Sequential(*[ConvNextBlock(dims[2],
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[2])])
        

        self.down4 = DownsampleBlock(dims[2], dims[3], stride=strides[2], kernel_size=kernels[2], padding=padding[2])
        self.stage4 = nn.Sequential(*[ConvNextBlock(dims[3],
                                                    drop_path=cfg.convnext_drop_path_rate,
                                                    layer_scale_init_value=cfg.convnext_layer_scale_init_value) 
                                      for _ in range(counts[3])])
        
        self._output_channels = dims[3]
        
        # Calculate width reduction factor
        # Stem stride (4) * downsample strides X components
        self.width_downsample_factor = cfg.stem_stride
        for s in strides:
            ts = s if isinstance(s, (tuple, list)) else (s, s)
            self.width_downsample_factor *= ts[1]

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def forward(self, x: Float[Tensor, "batch 3 80 width"]) -> Float[Tensor, "batch 512 2 width_sub"]:
        # Encoding stages
        x = self.stem(x)        # [B, 64, H/4, W/4]
        x = self.stage1(x)
        
        x = self.down2(x)
        x = self.stage2(x)
        
        x = self.down3(x)
        x = self.stage3(x)
        
        x = self.down4(x)
        x = self.stage4(x)
        
        return x
    
    def get_downsample_factor(self) -> int:
        """Returns the total width downsample factor."""
        return self.width_downsample_factor


# ========== REGISTRATION ==========

REGISTRY.register_encoder(
    name='convnext',
    cls=ConvNextEncoder,
    description='ConvNeXt based image encoder',
    compatible_heights=[80],
    width_divisor=4,
)

REGISTRY.register_encoder(
    name='resnet',
    cls=ResNetEncoder,
    description='ResNet based image encoder',
    compatible_heights=[80],
    width_divisor=4,
)

REGISTRY.register_encoder(
    name='convnext_separated',
    cls=ConvNextSeparatedEncoder,
    description='ConvNeXt separated based image encoder',
    compatible_heights=[80],
    width_divisor=4,
)
