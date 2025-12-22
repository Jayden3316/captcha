"""
Base classes for all model components.
Defines clear interfaces that all components must implement.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn
from jaxtyping import Float

Tensor = torch.Tensor

class BaseImageEncoder(nn.Module, ABC):
    """
    Base class for all image backbones.
    
    Encoders transform raw images into spatial feature maps:
    [B, C_in, H, W] -> [B, C_out, H', W']
    """
    
    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of output channels (C_out)."""
        pass
    
    @abstractmethod
    def forward(self, image: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch out_channel out_height out_width"]:
        """
        Encode image to spatial features.
        
        Args:
            image: Input image [B, C, H, W]
            
        Returns:
            features: Spatial feature map [B, C_out, H', W']
        """
        pass


class BaseAdapter(nn.Module, ABC):
    """
    Base class for adapters that bridge Encoders and Heads/Projectors.
    
    Adapters transform spatial features into task-specific formats.
    """
    
    def __init__(self, input_channels: int):
        super().__init__()
        self.input_channels = input_channels
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the output features (last dimension)."""
        pass


class BaseGenerationAdapter(BaseAdapter):
    """
    Adapter for generation tasks (Sequence output).
    
    [B, C, H, W] -> [B, Seq, Dim]
    """
    
    @abstractmethod
    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch seq dim"]:
        pass
        
    @abstractmethod
    def get_sequence_length(self, input_width: int) -> int:
        """Calculate expected sequence length given input width."""
        pass


class BaseClassificationAdapter(BaseAdapter):
    """
    Adapter for classification tasks (Fixed vector output).
    
    [B, C, H, W] -> [B, Dim]
    """
    
    @abstractmethod
    def forward(self, x: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch dim"]:
        pass


class BaseProjector(nn.Module, ABC):
    """
    Base class for dimension projectors.
    
    Projectors bridge encoder output to transformer input:
    [B, Seq, EncDim] -> [B, Seq, d_model]
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: Float[Tensor, "batch seq input_dim"]) -> Float[Tensor, "batch seq output_dim"]:
        """
        Project features to target dimension.
        
        Args:
            x: Input features [B, Seq, EncDim]
            
        Returns:
            projected: Projected features [B, Seq, d_model]
        """
        pass


class BaseSequenceModel(nn.Module, ABC):
    """
    Base class for sequence processing models.
    
    Sequence models process token sequences:
    [B, Seq, d_model] -> [B, Seq, d_model]
    
    May optionally use cross-attention to encoder outputs.
    """
    
    @property
    @abstractmethod
    def requires_cross_attention(self) -> bool:
        """Whether this model needs encoder outputs for cross-attention."""
        pass
    
    @abstractmethod
    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"],
        encoder_out: Optional[Float[Tensor, "batch enc_seq d_model"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        """
        Process sequence, optionally with cross-attention.
        
        Args:
            x: Input sequence [B, Seq, d_model]
            encoder_out: Optional encoder outputs for cross-attention [B, EncSeq, d_model]
            
        Returns:
            output: Processed sequence [B, Seq, d_model]
        """
        pass


class BaseHead(nn.Module, ABC):
    """
    Base class for output heads.
    
    Heads convert sequence representations to final outputs:
    [B, Seq, d_model] -> [B, Seq, VocabSize] or [B, NumClasses]
    """
    
    @property
    @abstractmethod
    def decoding_type(self) -> str:
        """
        Return the decoding strategy this head uses.
        
        Returns:
            One of: 'ctc', 'autoregressive', 'parallel', 'classification'
        """
        pass
    
    @property
    @abstractmethod
    def output_shape_info(self) -> dict:
        """
        Return information about output shape.
        
        Returns:
            Dict with keys:
                - 'type': 'sequence' or 'single'
                - 'size': output vocabulary/class size
                - 'includes_blank': whether vocab includes CTC blank token
        """
        pass
    
    @abstractmethod
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Tensor:
        """
        Generate output logits.
        
        Args:
            x: Sequence features [B, Seq, d_model]
            
        Returns:
            logits: Output logits
                For sequence outputs: [B, Seq, VocabSize]
                For classification: [B, NumClasses]
        """
        pass