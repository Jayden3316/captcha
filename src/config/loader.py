from typing import Dict, Any
from src.config.config import (
    ExperimentConfig, DatasetConfig, TrainingConfig, ModelConfig,
    AsymmetricConvNextEncoderConfig, ResNetEncoderConfig,
    LinearProjectorConfig, MLPProjectorConfig, IdentityProjectorConfig, BottleneckProjectorConfig, ResidualProjectorConfig,
    TransformerEncoderConfig, TransformerDecoderConfig, RNNConfig, BiLSTMConfig,
    LinearHeadConfig, MLPHeadConfig, ClassificationHeadConfig,
    FlattenAdapterConfig, VerticalFeatureAdapterConfig, GlobalPoolingAdapterConfig, SequencePoolingAdapterConfig
)

def hydrate_config(data: Dict[str, Any]) -> ExperimentConfig:
    """Manually hydrate dictionary into ExperimentConfig hierarchy.
    Shared logic extracted from cli.py.
    """
    
    # 1. Dataset Config
    ds_data = data.get('dataset_config', data.get('dataset', {}))
    dataset_config = DatasetConfig(**ds_data)
    
    # 2. Training Config
    tr_data = data.get('training_config', data.get('training', {}))
    training_config = TrainingConfig(**tr_data)
    
    # 3. Model Config
    mc_data = data.get('model_config', data.get('model', {}))
    
    # Helper to selecting config class based on type name
    def get_config_obj(type_name, config_dict, mapping):
        # If type_name is missing, return None (no implicit default)
        if not type_name:
             return None
        
        # If type_name is present but unknown, raise Error (no implicit default)
        cls = mapping.get(type_name)
        if cls is None:
            raise ValueError(f"Unknown config type: '{type_name}'. Available: {list(mapping.keys())}")
        
        # If config_dict is None, treat as empty dict
        if config_dict is None:
            config_dict = {}

        # Filter keys that valid for the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_args = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_args)

    # Encoders
    encoder_type = mc_data.get('encoder_type')
    encoder_cls_map = {
        'asymmetric_convnext': AsymmetricConvNextEncoderConfig,
        'resnet': ResNetEncoderConfig
    }
    encoder_config = get_config_obj(encoder_type, mc_data.get('encoder_config'), encoder_cls_map)

    # Adapters
    adapter_type = mc_data.get('adapter_type')
    adapter_cls_map = {
        'flatten': FlattenAdapterConfig,        
        'vertical_feature': VerticalFeatureAdapterConfig,
        'global_pool': GlobalPoolingAdapterConfig
    }

    if adapter_type:
        adapter_config = get_config_obj(adapter_type, mc_data.get('adapter_config'), adapter_cls_map)
    else:
        # If adapter_type is missing, let ModelConfig handle defaults entirely
        # UNLESS the user provided adapter_config block without type? Rare.
        adapter_config = None

    # Projectors
    proj_type = mc_data.get('projector_type')
    proj_cls_map = {
        'linear': LinearProjectorConfig,
        'mlp': MLPProjectorConfig,
        'identity': IdentityProjectorConfig,
        'bottleneck': BottleneckProjectorConfig,
        'residual': ResidualProjectorConfig
    }
    projector_config = get_config_obj(proj_type, mc_data.get('projector_config'), proj_cls_map)
    
    # Sequence Models
    seq_type = mc_data.get('sequence_model_type')
    seq_cls_map = {
        'transformer_encoder': TransformerEncoderConfig,
        'transformer_decoder': TransformerDecoderConfig, 
        'transformer_decoder_detr': TransformerDecoderConfig, # Alias
        'rnn': RNNConfig,
        'bilstm': BiLSTMConfig
    }
    sequence_model_config = get_config_obj(seq_type, mc_data.get('sequence_model_config'), seq_cls_map)

    # Heads
    head_type = mc_data.get('head_type')
    head_cls_map = {
        'linear': LinearHeadConfig, # and 'ctc' shares this
        'ctc': LinearHeadConfig,
        'mlp': MLPHeadConfig,
        'classification': ClassificationHeadConfig
    }
    head_config = get_config_obj(head_type, mc_data.get('head_config'), head_cls_map)
    
    # Model Config Args
    # We construct args map to pass only what's in YAML, avoiding Loader-level defaults
    # defaulting is handled by ModelConfig.__post_init__ or field defaults
    # Adapters
    adapter_type = mc_data.get('adapter_type')
    encoder_adapter_type = mc_data.get('encoder_adapter_type')
    sequence_adapter_type = mc_data.get('sequence_adapter_type')
    
    adapter_cls_map = {
        'flatten': FlattenAdapterConfig,
        'vertical_feature': VerticalFeatureAdapterConfig,
        'global_pool': GlobalPoolingAdapterConfig,
        'sequence_pool': SequencePoolingAdapterConfig
    }
    
    if adapter_type:
        adapter_config = get_config_obj(adapter_type, mc_data.get('adapter_config'), adapter_cls_map)
    else:
        adapter_config = None
        
    if encoder_adapter_type:
        encoder_adapter_config = get_config_obj(encoder_adapter_type, mc_data.get('encoder_adapter_config'), adapter_cls_map)
    else:
         encoder_adapter_config = None
         
    if sequence_adapter_type:
        sequence_adapter_config = get_config_obj(sequence_adapter_type, mc_data.get('sequence_adapter_config'), adapter_cls_map)
    else:
         sequence_adapter_config = None

    # ... (Projectors, Sequence Models, Heads logic remains)

    # Model Config Args
    mc_args = {
        'encoder_type': encoder_type,
        'encoder_config': encoder_config,
        'adapter_type': adapter_type,
        'adapter_config': adapter_config,
        'encoder_adapter_type': encoder_adapter_type,
        'encoder_adapter_config': encoder_adapter_config,
        'sequence_adapter_type': sequence_adapter_type,
        'sequence_adapter_config': sequence_adapter_config,
        'projector_type': proj_type,
        'projector_config': projector_config,
        'sequence_model_type': seq_type,
        'sequence_model_config': sequence_model_config,
        'head_type': head_type,
        'head_config': head_config,
        'pipeline_type': mc_data.get('pipeline_type'),
    }
    
    # Optional scalar fields
    for field in ['d_model', 'd_vocab', 'loss_type', 'task_type']:
        if field in mc_data:
            mc_args[field] = mc_data[field]

    model_config = ModelConfig(**mc_args)
    
    
    # Construct ExperimentConfig with optional fields
    # We only pass fields that are present in the data to allow dataclass defaults to take effect
    config_args = {
        'experiment_name': data.get('experiment_name', 'custom_run'),
        'dataset_config': dataset_config,
        'training_config': training_config,
        'model_config': model_config,
    }
    
    # Optional fields that should only be passed if present in YAML
    optional_fields = [
        'metadata_path', 
        'train_metadata_path', 
        'val_metadata_path', 
        'image_base_dir', 
        'seed'
    ]
    
    for field_name in optional_fields:
        if field_name in data:
            config_args[field_name] = data[field_name]

    return ExperimentConfig(**config_args)
