"""
Central registry for all model components.
Provides discovery, validation, and factory methods.
"""
from typing import Type, Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from .components.base import BaseImageEncoder, BaseAdapter, BaseProjector, BaseSequenceModel, BaseHead


@dataclass
class ComponentMetadata:
    """Metadata about a registered component."""
    name: str
    cls: Type
    description: str = ""
    compatible_heights: List[int] = field(default_factory=list)
    width_divisor: int = 1
    width_bias: int = 0
    requires: List[str] = field(default_factory=list)  # Dependencies
    provides: List[str] = field(default_factory=list)  # Capabilities
    extra: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """
    Central registry for all model components.
    
    Supports:
    - Component registration and discovery
    - Automatic compatibility validation
    - Factory methods for component creation
    - Documentation generation
    """
    
    def __init__(self):
        self._encoders: Dict[str, ComponentMetadata] = {}
        self._adapters: Dict[str, ComponentMetadata] = {}
        self._projectors: Dict[str, ComponentMetadata] = {}
        self._sequence_models: Dict[str, ComponentMetadata] = {}
        self._heads: Dict[str, ComponentMetadata] = {}
        
    # ========== ENCODERS ==========
    
    def register_encoder(
        self, 
        name: str, 
        cls: Type[BaseImageEncoder],
        description: str = "",
        compatible_heights: List[int] = None,
        width_divisor: int = 1,
        width_bias: int = 0,
        **extra
    ):
        """
        Register an encoder component.
        
        Args:
            name: Unique identifier for this encoder
            cls: Encoder class (must inherit from BaseImageEncoder)
            description: Human-readable description
            compatible_heights: List of supported input heights (empty = any)
            width_divisor: Input width must be divisible by this value
            **extra: Additional metadata
        """
        if not issubclass(cls, BaseImageEncoder):
            raise ValueError(f"{cls} must inherit from BaseImageEncoder")
            
        self._encoders[name] = ComponentMetadata(
            name=name,
            cls=cls,
            description=description,
            compatible_heights=compatible_heights or [],
            width_divisor=width_divisor,
            width_bias=width_bias,
            extra=extra
        )
        
    def get_encoder(self, name: str) -> Type[BaseImageEncoder]:
        """Get encoder class by name."""
        if name not in self._encoders:
            available = ', '.join(self._encoders.keys())
            raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
        return self._encoders[name].cls
    
    def list_encoders(self) -> Dict[str, Dict[str, Any]]:
        """List all registered encoders with their metadata."""
        return {
            name: {
                'description': meta.description,
                'compatible_heights': meta.compatible_heights,
                'width_divisor': meta.width_divisor,
                'width_bias': meta.width_bias,
                **meta.extra
            }
            for name, meta in self._encoders.items()
        }

    # ========== ADAPTERS ==========
    
    def register_adapter(
        self,
        name: str,
        cls: Type[BaseAdapter],
        description: str = "",
        type: str = "generation", # 'generation' or 'classification'
        **extra
    ):
        """Register an adapter component."""
        if not issubclass(cls, BaseAdapter):
            raise ValueError(f"{cls} must inherit from BaseAdapter")
            
        self._adapters[name] = ComponentMetadata(
            name=name,
            cls=cls,
            description=description,
            extra={'type': type, **extra}
        )
        
    def get_adapter(self, name: str) -> Type[BaseAdapter]:
        """Get adapter class by name."""
        if name not in self._adapters:
            available = ', '.join(self._adapters.keys())
            raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
        return self._adapters[name].cls

    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all registered adapters."""
        return {
            name: {
                'description': meta.description,
                'type': meta.extra.get('type', 'unknown'),
                **meta.extra
            }
            for name, meta in self._adapters.items()
        }
    
    # ========== PROJECTORS ==========
    
    def register_projector(
        self,
        name: str,
        cls: Type[BaseProjector],
        description: str = "",
        **extra
    ):
        """Register a projector component."""
        if not issubclass(cls, BaseProjector):
            raise ValueError(f"{cls} must inherit from BaseProjector")
            
        self._projectors[name] = ComponentMetadata(
            name=name,
            cls=cls,
            description=description,
            extra=extra
        )
        
    def get_projector(self, name: str) -> Type[BaseProjector]:
        """Get projector class by name."""
        if name not in self._projectors:
            available = ', '.join(self._projectors.keys())
            raise ValueError(f"Unknown projector '{name}'. Available: {available}")
        return self._projectors[name].cls
    
    def list_projectors(self) -> Dict[str, Dict[str, Any]]:
        """List all registered projectors."""
        return {
            name: {
                'description': meta.description,
                **meta.extra
            }
            for name, meta in self._projectors.items()
        }
    
    # ========== SEQUENCE MODELS ==========
    
    def register_sequence_model(
        self,
        name: str,
        cls: Type[BaseSequenceModel],
        description: str = "",
        requires_cross_attention: bool = False,
        **extra
    ):
        """Register a sequence model component."""
        if not issubclass(cls, BaseSequenceModel):
            raise ValueError(f"{cls} must inherit from BaseSequenceModel")
            
        self._sequence_models[name] = ComponentMetadata(
            name=name,
            cls=cls,
            description=description,
            extra={'requires_cross_attention': requires_cross_attention, **extra}
        )
        
    def get_sequence_model(self, name: str) -> Type[BaseSequenceModel]:
        """Get sequence model class by name."""
        if name not in self._sequence_models:
            available = ', '.join(self._sequence_models.keys())
            raise ValueError(f"Unknown sequence model '{name}'. Available: {available}")
        return self._sequence_models[name].cls
    
    def list_sequence_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered sequence models."""
        return {
            name: {
                'description': meta.description,
                **meta.extra
            }
            for name, meta in self._sequence_models.items()
        }
    
    # ========== HEADS ==========
    
    def register_head(
        self,
        name: str,
        cls: Type[BaseHead],
        description: str = "",
        decoding_type: str = "",
        compatible_losses: List[str] = None,
        **extra
    ):
        """Register a head component."""
        if not issubclass(cls, BaseHead):
            raise ValueError(f"{cls} must inherit from BaseHead")
            
        self._heads[name] = ComponentMetadata(
            name=name,
            cls=cls,
            description=description,
            extra={
                'decoding_type': decoding_type,
                'compatible_losses': compatible_losses or [],
                **extra
            }
        )
        
    def get_head(self, name: str) -> Type[BaseHead]:
        """Get head class by name."""
        if name not in self._heads:
            available = ', '.join(self._heads.keys())
            raise ValueError(f"Unknown head '{name}'. Available: {available}")
        return self._heads[name].cls
    
    def list_heads(self) -> Dict[str, Dict[str, Any]]:
        """List all registered heads."""
        return {
            name: {
                'description': meta.description,
                **meta.extra
            }
            for name, meta in self._heads.items()
        }
    
    # ========== VALIDATION ==========
    
    def validate_config(self, config) -> List[str]:
        """
        Validate that a configuration's components are compatible.
        
        Args:
            config: Configuration object with component types
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check encoder exists and height compatibility
        if hasattr(config, 'encoder_type'):
            if config.encoder_type not in self._encoders:
                errors.append(f"Unknown encoder: {config.encoder_type}")
            else:
                encoder_meta = self._encoders[config.encoder_type]
                if encoder_meta.compatible_heights:
                    if config.input_height not in encoder_meta.compatible_heights:
                        errors.append(
                            f"Input height {config.input_height} incompatible with "
                            f"{config.encoder_type}. Supported heights: "
                            f"{encoder_meta.compatible_heights}"
                        )
                        
                # Check width divisibility
                if hasattr(config, 'input_width') and config.input_width:
                    if (config.input_width - encoder_meta.width_bias) % encoder_meta.width_divisor != 0:
                        formula = f"{encoder_meta.width_divisor}k"
                        if encoder_meta.width_bias != 0:
                            formula += f" + {encoder_meta.width_bias}"
                            
                        errors.append(
                            f"Input width {config.input_width} incompatible with {config.encoder_type}. "
                            f"Must satisfy width = {formula}"
                        )
        
        # Check projector exists
        if hasattr(config, 'projector_type'):
            if config.projector_type not in self._projectors:
                errors.append(f"Unknown projector: {config.projector_type}")
        
        # Check sequence model exists
        if hasattr(config, 'sequence_model_type'):
            if config.sequence_model_type not in self._sequence_models:
                errors.append(f"Unknown sequence model: {config.sequence_model_type}")
        
        # Check head exists
        if hasattr(config, 'head_type'):
            if config.head_type not in self._heads:
                errors.append(f"Unknown head: {config.head_type}")
            else:
                head_meta = self._heads[config.head_type]
                
                # Check loss compatibility
                if hasattr(config, 'loss_type'):
                    compatible_losses = head_meta.extra.get('compatible_losses', [])
                    if compatible_losses and config.loss_type not in compatible_losses:
                        errors.append(
                            f"Head {config.head_type} incompatible with loss "
                            f"{config.loss_type}. Compatible losses: {compatible_losses}"
                        )
        
        return errors
    
    # ========== UTILITIES ==========
    
    def print_summary(self):
        """Print a summary of all registered components."""
        print("=" * 60)
        print("REGISTERED COMPONENTS")
        print("=" * 60)
        
        print("\n ENCODERS:")
        for name, meta in self._encoders.items():
            print(f"  • {name}")
            if meta.description:
                print(f"    {meta.description}")
            if meta.compatible_heights:
                print(f"    Heights: {meta.compatible_heights}")
            print(f"    Width divisor: {meta.width_divisor}")
            if meta.width_bias != 0:
                print(f"    Width bias: {meta.width_bias}")
        
        print("\n PROJECTORS:")
        for name, meta in self._projectors.items():
            print(f"  • {name}")
            if meta.description:
                print(f"    {meta.description}")
        
        print("\n SEQUENCE MODELS:")
        for name, meta in self._sequence_models.items():
            print(f"  • {name}")
            if meta.description:
                print(f"    {meta.description}")
            cross_attn = meta.extra.get('requires_cross_attention', False)
            print(f"    Cross-attention: {cross_attn}")
        
        print("\n HEADS:")
        for name, meta in self._heads.items():
            print(f"  • {name}")
            if meta.description:
                print(f"    {meta.description}")
            decoding = meta.extra.get('decoding_type', 'unknown')
            print(f"    Decoding: {decoding}")
            losses = meta.extra.get('compatible_losses', [])
            if losses:
                print(f"    Compatible losses: {losses}")
        
        print("=" * 60)


# Global singleton registry
REGISTRY = ComponentRegistry()