"""
Modular Configuration System for Captcha Architecture.

Hierarchy:
    ExperimentConfig
    ├── DatasetConfig
    ├── ModelConfig
    │   ├── EncoderConfig (type-specific)
    │   ├── ProjectorConfig (type-specific)
    │   ├── SequenceModelConfig (type-specific)
    │   └── HeadConfig (type-specific)
    └── TrainingConfig

All hyperparameters are first-class citizens with no hardcoded values in components.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Callable
from enum import Enum
import torch
from transformer_lens import HookedTransformerConfig


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class ActivationType(str, Enum):
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    TANH = "tanh"


class NormalizationType(str, Enum):
    RMS = "RMS"
    LAYER = "layer"
    BATCH = "batch"


class PositionalEmbeddingType(str, Enum):
    ROTARY = "rotary"
    STANDARD = "standard"
    NONE = "none"


class AttentionDirection(str, Enum):
    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"


class PoolingType(str, Enum):
    MEAN = "mean"
    MAX = "max"
    FIRST = "first"


class LossType(str, Enum):
    CTC = "ctc"
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class TaskType(str, Enum):
    GENERATION = "generation"
    CLASSIFICATION = "classification"


# ============================================================================
# DATASET CONFIG
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation and processing."""
    
    # Image dimensions
    width: int = 200
    height: int = 80
    
    # Font configuration
    fonts: Optional[List[str]] = None
    font_sizes: Optional[List[int]] = None
    max_fonts_per_family: int = 2
    
    # Content configuration
    word_path: Optional[str] = None
    
    # Noise configuration
    noise_bg_density: int = 5000
    add_noise_dots: bool = True
    add_noise_curve: bool = True
    
    # Spacing configuration
    extra_spacing: int = -5
    spacing_jitter: int = 6
    
    # Color configuration
    bg_color: Optional[tuple[int, int, int]] = None
    fg_color: Optional[tuple[int, int, int, int]] = None
    
    # Output configuration
    image_ext: str = "png"
    
    # Text transformation
    word_transform: Optional[str] = None  # e.g., "random_capitalize"
    random_capitalize: bool = True
    
    # Fine-grained Captcha Distortion (Captcha Library Defaults overrides)
    character_offset_dx: Optional[tuple[int, int]] = (0, 4)
    character_offset_dy: Optional[tuple[int, int]] = (0, 6)
    character_rotate: Optional[tuple[int, int]] = (-30, 30)
    character_warp_dx: Optional[tuple[float, float]] = (0.1, 0.3)
    character_warp_dy: Optional[tuple[float, float]] = (0.2, 0.3)
    word_space_probability: float = 0.5
    word_offset_dx: float = 0.25
    
    # Processing configuration
    target_height: int = 80  # For resizing
    width_divisor: int = 4   # Width must be divisible by this
    width_bias: int = 0      # Bias for width formula (width = divisor * k + bias)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width,
            'height': self.height,
            'noise_bg_density': self.noise_bg_density,
            'add_noise_dots': self.add_noise_dots,
            'add_noise_curve': self.add_noise_curve,
            'extra_spacing': self.extra_spacing,
            'spacing_jitter': self.spacing_jitter,
            'target_height': self.target_height,
            'width_divisor': self.width_divisor,
            'width_bias': self.width_bias,
        }


# ============================================================================
# ENCODER CONFIGS
# ============================================================================

@dataclass
class AsymmetricConvNextEncoderConfig:
    """Configuration for AsymmetricConvNextEncoder."""
    
    # Stage dimensions
    dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Stage block counts (per stage, in order)
    stage_block_counts: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    
    # Stem configuration
    stem_kernel_size: int = 4
    stem_stride: int = 4
    stem_in_channels: int = 3
    
    # Downsample configuration
    downsample_kernel_size: tuple[int, int] = (2, 1)
    downsample_stride: tuple[int, int] = (2, 1)
    
    # ConvNextBlock configuration
    convnext_kernel_size: int = 7
    convnext_drop_path_rate: float = 0.0
    convnext_layer_scale_init_value: float = 1e-6
    convnext_activation: ActivationType = ActivationType.GELU
    convnext_expansion_ratio: int = 4  # MLP expansion in ConvNextBlock
    
    # Normalization
    norm_eps: float = 1e-6
    
    # Output dimension (auto-computed from dims[-1] if not specified)
    output_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.dims[-1]
        if len(self.stage_block_counts) != len(self.dims):
            raise ValueError(f"stage_block_counts length ({len(self.stage_block_counts)}) must match dims length ({len(self.dims)})")


@dataclass
class LegacyCNNEncoderConfig:
    """Configuration for LegacyCNNEncoder."""
    
    # Convolution layers configuration
    filter_sizes: List[int] = field(default_factory=lambda: [7, 5])
    strides: List[int] = field(default_factory=lambda: [1, 1])
    channels: List[int] = field(default_factory=lambda: [16, 32])
    in_channels: int = 3
    
    # Pooling configuration
    pool_kernel_size: int = 2
    pool_stride: int = 2
    
    # Patch extraction configuration
    patch_height: int = 14
    patch_width: int = 7
    patch_stride: int = 7
    
    # Activation
    activation: ActivationType = ActivationType.SILU
    
    # Output dimension (auto-computed as channels[-1] * patch_height * patch_width)
    output_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.channels[-1] * self.patch_height * self.patch_width
        if len(set([len(self.filter_sizes), len(self.strides), len(self.channels)])) != 1:
            raise ValueError("filter_sizes, strides, and channels must have same length")


# Union type for encoder configs
EncoderConfig = Union[AsymmetricConvNextEncoderConfig, LegacyCNNEncoderConfig]


# ============================================================================
# PROJECTOR CONFIGS
# ============================================================================

@dataclass
class LinearProjectorConfig:
    """Configuration for LinearProjector."""
    pass  # No extra parameters beyond input_dim/output_dim

@dataclass
class IdentityProjectorConfig:
    """Configuration for IdentityProjector."""
    pass  # No extra parameters beyond input_dim/output_dim

@dataclass
class MLPProjectorConfig:
    """Configuration for MLPProjector."""
    
    hidden_dim: Optional[int] = None  # If None, computed as (input_dim + output_dim) // 2
    num_layers: int = 2
    activation: ActivationType = ActivationType.GELU


@dataclass
class BottleneckProjectorConfig:
    """Configuration for BottleneckProjector."""
    
    bottleneck_dim: Optional[int] = None  # If None, computed as min(input_dim, output_dim) // 2
    activation: ActivationType = ActivationType.GELU


@dataclass
class ResidualProjectorConfig:
    """Configuration for ResidualProjector."""
    
    hidden_dim: Optional[int] = None  # If None, computed as (input_dim + output_dim) // 2
    activation: ActivationType = ActivationType.GELU


# Union type for projector configs
ProjectorConfig = Union[
    LinearProjectorConfig,
    MLPProjectorConfig,
    BottleneckProjectorConfig,
    ResidualProjectorConfig
]


# ============================================================================
# SEQUENCE MODEL CONFIGS
# ============================================================================

@dataclass
class TransformerConfig(HookedTransformerConfig):
    """
    Base configuration for Transformer-based sequence models.
    Inherits from HookedTransformerConfig to ensure compatibility.
    """
    
    # Override defaults
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    d_mlp: int = 1024
    n_ctx: int = 384
    d_vocab: int = 62
    d_head: Optional[int] = None
    act_fn: str = "gelu" # Required by HookedTransformerConfig if attn_only=False
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None
    
    # Extra fields for our usage
    dropout: float = 0.0
    attn_dropout: float = 0.0
    
    def __post_init__(self):
        # Ensure d_head is computed if not set
        # HookedTransformerConfig's __post_init__ MIGHT handle d_head computation, 
        # but evidently it crashes if n_params calculation happens before that.
        # So we force it here.
        if self.d_head is None and self.d_model is not None and self.n_heads is not None:
             self.d_head = self.d_model // self.n_heads
        
        # Run parent post init
        super().__post_init__()
        
        # Ensure enums are strings if they were passed as Enums (just in case)
        # These fields are now part of HookedTransformerConfig, but we keep the conversion
        # logic in case they are set with Enum types directly.
        if hasattr(self.attention_dir, 'value'): self.attention_dir = self.attention_dir.value
        if hasattr(self.act_fn, 'value'): self.act_fn = self.act_fn.value
        if hasattr(self.positional_embedding_type, 'value'): self.positional_embedding_type = self.positional_embedding_type.value
        if hasattr(self.normalization_type, 'value'): self.normalization_type = self.normalization_type.value


@dataclass
class TransformerEncoderConfig(TransformerConfig):
    """Configuration for TransformerEncoderModel."""
    pass  # Uses base TransformerConfig


@dataclass
class TransformerDecoderConfig(TransformerConfig):
    """Configuration for TransformerDecoderModel (with cross-attention)."""
    pass  # Uses base TransformerConfig, cross-attention is implied


@dataclass
class RNNConfig:
    """Configuration for RNNSequenceModel."""
    
    num_layers: int = 2
    hidden_size: int = 256  # Should match d_model from pipeline
    dropout: float = 0.1
    bidirectional: bool = False
    activation: ActivationType = ActivationType.RELU  # Applied after RNN output


@dataclass
class BiLSTMConfig:
    """Configuration for BiLSTMSequenceModel."""
    
    num_layers: int = 2
    hidden_size: int = 128  # Per direction, total output will be hidden_size * 2
    dropout: float = 0.1
    # Note: bidirectional=True is implicit for BiLSTM


# Union type for sequence model configs
SequenceModelConfig = Union[
    TransformerEncoderConfig,
    TransformerDecoderConfig,
    RNNConfig,
    BiLSTMConfig
]


# ============================================================================
# HEAD CONFIGS
# ============================================================================

@dataclass
class LinearHeadConfig:
    """Configuration for LinearHead."""
    d_model: Optional[int] = None
    d_vocab: Optional[int] = None
    head_type: Optional[str] = None
    loss_type: Optional[str] = None


@dataclass
class MLPHeadConfig:
    """Configuration for MLPHead."""
    
    num_layers: int = 2
    hidden_dim: Optional[int] = None  # If None, computed as d_model * 2
    dropout: float = 0.1
    activation: ActivationType = ActivationType.GELU
    d_model: Optional[int] = None
    d_vocab: Optional[int] = None
    head_type: Optional[str] = None
    loss_type: Optional[str] = None


@dataclass
class ClassificationHeadConfig:
    """Configuration for ClassificationHead."""
    
    num_classes: int = 10
    pooling_type: PoolingType = PoolingType.MEAN
    hidden_dim: Optional[int] = None  # If None, computed as d_model
    activation: ActivationType = ActivationType.GELU
    dropout: float = 0.1
    d_model: Optional[int] = None
    d_vocab: Optional[int] = None
    head_type: Optional[str] = None
    loss_type: Optional[str] = None


# Union type for head configs
HeadConfig = Union[LinearHeadConfig, MLPHeadConfig, ClassificationHeadConfig]


# ============================================================================
# MODEL CONFIG
# ============================================================================

@dataclass
class ModelConfig:
    """Main model configuration consolidating all component configs."""
    
    # Component type selection
    encoder_type: str = 'asymmetric_convnext'
    projector_type: str = 'linear'
    sequence_model_type: str = 'transformer_encoder'
    head_type: str = 'ctc'
    task_type: TaskType = TaskType.GENERATION
    
    # Component-specific configs
    encoder_config: Optional[EncoderConfig] = None
    projector_config: Optional[ProjectorConfig] = None
    sequence_model_config: Optional[SequenceModelConfig] = None
    head_config: Optional[HeadConfig] = None
    
    # Vocabulary
    d_vocab: int = 62  # Number of characters (excluding blank/pad)
    
    # Loss type
    loss_type: LossType = LossType.CTC
    
    # Pipeline dimensions (auto-resolved if None)
    d_model: Optional[int] = None  # If None, inherited from sequence_model_config or default 256
    
    def __post_init__(self):
        # Auto-create configs if not provided, using defaults
        if self.encoder_config is None:
            if self.encoder_type == 'asymmetric_convnext':
                self.encoder_config = AsymmetricConvNextEncoderConfig()
            elif self.encoder_type == 'legacy_cnn':
                self.encoder_config = LegacyCNNEncoderConfig()
            else:
                raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        
        if self.projector_config is None:
            if self.projector_type == 'linear':
                self.projector_config = LinearProjectorConfig()
            elif self.projector_type == 'mlp':
                self.projector_config = MLPProjectorConfig()
            elif self.projector_type == 'bottleneck':
                self.projector_config = BottleneckProjectorConfig()
            elif self.projector_type == 'residual':
                self.projector_config = ResidualProjectorConfig()
            elif self.projector_type == 'identity':
                self.projector_config = LinearProjectorConfig()  # Identity handled separately
            else:
                raise ValueError(f"Unknown projector_type: {self.projector_type}")
        
        if self.sequence_model_config is None:
            if self.sequence_model_type == 'transformer_encoder':
                self.sequence_model_config = TransformerEncoderConfig()
            elif self.sequence_model_type == 'transformer_decoder':
                self.sequence_model_config = TransformerDecoderConfig()
            elif self.sequence_model_type == 'rnn':
                self.sequence_model_config = RNNConfig()
            elif self.sequence_model_type == 'bilstm':
                self.sequence_model_config = BiLSTMConfig()
            else:
                raise ValueError(f"Unknown sequence_model_type: {self.sequence_model_type}")
        
        if self.head_config is None:
            if self.head_type == 'linear' or self.head_type == 'ctc':
                self.head_config = LinearHeadConfig()
            elif self.head_type == 'mlp':
                self.head_config = MLPHeadConfig()
            elif self.head_type == 'classification':
                self.head_config = ClassificationHeadConfig()
            else:
                raise ValueError(f"Unknown head_type: {self.head_type}")
        
        # Resolve d_model: use from sequence_model_config if available, else default
        if self.d_model is None:
            if isinstance(self.sequence_model_config, (TransformerEncoderConfig, TransformerDecoderConfig)):
                self.d_model = self.sequence_model_config.d_model
            elif isinstance(self.sequence_model_config, RNNConfig):
                self.d_model = self.sequence_model_config.hidden_size
            elif isinstance(self.sequence_model_config, BiLSTMConfig):
                self.d_model = self.sequence_model_config.hidden_size * 2  # Bidirectional
            else:
                self.d_model = 256  # Default
        
        # Sync d_model to sequence model config if needed
        if isinstance(self.sequence_model_config, RNNConfig):
            self.sequence_model_config.hidden_size = self.d_model
        elif isinstance(self.sequence_model_config, BiLSTMConfig):
            # BiLSTM hidden_size should be half of d_model
            if self.d_model % 2 != 0:
                raise ValueError(f"d_model ({self.d_model}) must be even for BiLSTM")
            self.sequence_model_config.hidden_size = self.d_model // 2

        self.validate_consistency()

    def validate_consistency(self):
        """Validate consistency of configurations."""
        # 1. Check d_model consistency
        if self.sequence_model_config and hasattr(self.sequence_model_config, 'd_model'):
            if self.sequence_model_config.d_model != self.d_model:
                raise ValueError(
                    f"Mismatch in d_model: ModelConfig has {self.d_model}, "
                    f"but SequenceModelConfig has {self.sequence_model_config.d_model}"
                )
        
        if self.head_config and hasattr(self.head_config, 'd_model') and self.head_config.d_model is not None:
             if self.head_config.d_model != self.d_model:
                raise ValueError(
                    f"Mismatch in d_model: ModelConfig has {self.d_model}, "
                    f"but HeadConfig has {self.head_config.d_model}"
                )

        # 2. Check d_vocab consistency
        if self.sequence_model_config and hasattr(self.sequence_model_config, 'd_vocab'):
             if self.sequence_model_config.d_vocab != self.d_vocab:
                 # Note: TransformerLens config usually requires d_vocab
                raise ValueError(
                    f"Mismatch in d_vocab: ModelConfig has {self.d_vocab}, "
                    f"but SequenceModelConfig has {self.sequence_model_config.d_vocab}"
                )

        if self.head_config and hasattr(self.head_config, 'd_vocab') and self.head_config.d_vocab is not None:
             if self.head_config.d_vocab != self.d_vocab:
                raise ValueError(
                    f"Mismatch in d_vocab: ModelConfig has {self.d_vocab}, "
                    f"but HeadConfig has {self.head_config.d_vocab}"
                )

    
    def get_encoder_output_dim(self) -> int:
        """Get encoder output dimension."""
        if isinstance(self.encoder_config, AsymmetricConvNextEncoderConfig):
            return self.encoder_config.output_dim
        elif isinstance(self.encoder_config, LegacyCNNEncoderConfig):
            return self.encoder_config.output_dim
        else:
            raise ValueError(f"Unknown encoder_config type: {type(self.encoder_config)}")
    
    def get_projector_input_dim(self) -> int:
        """Get projector input dimension (encoder output)."""
        return self.get_encoder_output_dim()
    
    def get_projector_output_dim(self) -> int:
        """Get projector output dimension (sequence model input)."""
        return self.d_model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'encoder_type': self.encoder_type,
            'projector_type': self.projector_type,
            'sequence_model_type': self.sequence_model_type,
            'head_type': self.head_type,
            'd_vocab': self.d_vocab,
            'd_model': self.d_model,
            'loss_type': self.loss_type.value if isinstance(self.loss_type, LossType) else self.loss_type,
            'task_type': self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
        }


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Basic training parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    
    # Optimizer configuration
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler_type: Optional[str] = None  # e.g., "cosine", "step", "plateau"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Gradient configuration
    grad_clip_norm: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Device and parallelism
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_best_only: bool = False
    monitor_metric: str = "val_exact_match"  # Metric to monitor for best model
    
    # Logging
    wandb_project: str = "captcha-ocr"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 10
    
    # Validation
    val_split: float = 0.1  # Fraction of dataset to use for validation
    val_check_interval: float = 1.0  # Validate every N epochs
    
    # Data loading
    shuffle_train: bool = True
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'optimizer_type': self.optimizer_type.value if isinstance(self.optimizer_type, OptimizerType) else self.optimizer_type,
            'weight_decay': self.weight_decay,
            'grad_clip_norm': self.grad_clip_norm,
            'device': self.device,
        }


# ============================================================================
# EXPERIMENT CONFIG (MAIN CONFIG)
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Main experiment configuration consolidating all sub-configs.
    
    This is the single point of truth for all hyperparameters.
    """
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    seed: Optional[int] = None
    
    # Sub-configs
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    metadata_path: str = "validation_set/metadata.json"
    train_metadata_path: Optional[str] = None
    val_metadata_path: Optional[str] = None
    image_base_dir: str = "."
    
    def __post_init__(self):
        """Validate and sync configurations."""
        # Sync dataset height with model input height expectation
        # This ensures consistency between data generation and model input
        
        # If encoder expects specific height, update dataset config
        if isinstance(self.model_config.encoder_config, AsymmetricConvNextEncoderConfig):
            # AsymmetricConvNext expects height 80
            if self.dataset_config.height != 80:
                self.dataset_config.height = 80
                self.dataset_config.target_height = 80
                self.dataset_config.width_divisor = 4
        elif isinstance(self.model_config.encoder_config, LegacyCNNEncoderConfig):
            # Legacy CNN expects height 70 and specific width formula
            self.dataset_config.height = 70
            self.dataset_config.target_height = 70
            self.dataset_config.width_divisor = 28
            self.dataset_config.width_bias = 14
        
        # Sync RNN/LSTM hidden sizes with d_model if needed
        if isinstance(self.model_config.sequence_model_config, RNNConfig):
            self.model_config.sequence_model_config.hidden_size = self.model_config.d_model
        elif isinstance(self.model_config.sequence_model_config, BiLSTMConfig):
            if self.model_config.d_model % 2 != 0:
                raise ValueError("d_model must be even for BiLSTM")
            self.model_config.sequence_model_config.hidden_size = self.model_config.d_model // 2
        

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'dataset': self.dataset_config.to_dict(),
            'model': self.model_config.to_dict(),
            'training': self.training_config.to_dict(),
        }
    
    def describe(self) -> str:
        """Generate human-readable description."""
        return f"""
Experiment: {self.experiment_name}
{'=' * 60}

Dataset:
  Height: {self.dataset_config.height}
  Width divisor: {self.dataset_config.width_divisor}
  Width bias: {self.dataset_config.width_bias}

Model:
  Encoder: {self.model_config.encoder_type} (out_dim={self.model_config.get_encoder_output_dim()})
  Projector: {self.model_config.projector_type}
  Sequence: {self.model_config.sequence_model_type}
  Head: {self.model_config.head_type}
  d_model: {self.model_config.d_model}
  d_vocab: {self.model_config.d_vocab}

Training:
  Batch size: {self.training_config.batch_size}
  Epochs: {self.training_config.epochs}
  Learning rate: {self.training_config.learning_rate}
  Optimizer: {self.training_config.optimizer_type}
{'=' * 60}
        """