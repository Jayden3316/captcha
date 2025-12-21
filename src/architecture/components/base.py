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

class BaseEncoder(nn.Module, ABC):
    """
    Base class for all image encoders.
    
    Encoders transform images into token sequences:
    [B, C, H, W] -> [B, SeqLen, EncDim]
    """
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of encoder output features."""
        pass
    
    @abstractmethod
    def forward(self, image: Float[Tensor, "batch channel height width"]) -> Float[Tensor, "batch seq dim"]:
        """
        Encode image to token sequence.
        
        Args:
            image: Input image [B, C, H, W]
            
        Returns:
            features: Token sequence [B, SeqLen, EncDim]
        """
        pass
    
    def get_sequence_length(self, image_width: int) -> int:
        """
        Calculate output sequence length for given image width.
        Useful for validation and planning.
        
        Args:
            image_width: Width of input image
            
        Returns:
            Expected sequence length after encoding
        """
        raise NotImplementedError("Subclass should implement if known")


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