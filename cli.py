import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional

from src.config.config import (
    ExperimentConfig, DatasetConfig, TrainingConfig, ModelConfig,
    AsymmetricConvNextEncoderConfig, LegacyCNNEncoderConfig,
    LinearProjectorConfig, MLPProjectorConfig, IdentityProjectorConfig, BottleneckProjectorConfig, ResidualProjectorConfig,
    TransformerEncoderConfig, TransformerDecoderConfig, RNNConfig, BiLSTMConfig,
    LinearHeadConfig, MLPHeadConfig, ClassificationHeadConfig
)
from src.train import train

from generate_captchas import get_ttf_files, get_words, CaptchaGenerator, random_capitalize

def hydrate_config(data: dict) -> ExperimentConfig:
    """Manually hydrate dictionary into ExperimentConfig hierarchy."""
    
    # 1. Dataset Config
    ds_data = data.get('dataset_config', {})
    dataset_config = DatasetConfig(**ds_data)
    
    # 2. Training Config
    tr_data = data.get('training_config', {})
    training_config = TrainingConfig(**tr_data)
    
    # 3. Model Config
    mc_data = data.get('model_config', {})
    
    # Helper to selecting config class based on type name
    def get_config_obj(type_name, config_dict, mapping, default_cls):
        if not type_name or not config_dict:
            return default_cls()
        cls = mapping.get(type_name, default_cls)
        # Filter keys that valid for the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_args = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_args)

    # Encoders
    encoder_type = mc_data.get('encoder_type')
    encoder_cls_map = {
        'asymmetric_convnext': AsymmetricConvNextEncoderConfig,
        'legacy_cnn': LegacyCNNEncoderConfig
    }
    encoder_config = get_config_obj(encoder_type, mc_data.get('encoder_config'), encoder_cls_map, AsymmetricConvNextEncoderConfig)

    # Projectors
    proj_type = mc_data.get('projector_type')
    proj_cls_map = {
        'linear': LinearProjectorConfig,
        'mlp': MLPProjectorConfig,
        'identity': IdentityProjectorConfig,
        'bottleneck': BottleneckProjectorConfig,
        'residual': ResidualProjectorConfig
    }
    projector_config = get_config_obj(proj_type, mc_data.get('projector_config'), proj_cls_map, LinearProjectorConfig)
    
    # Sequence Models
    seq_type = mc_data.get('sequence_model_type')
    seq_cls_map = {
        'transformer_encoder': TransformerEncoderConfig,
        'transformer_decoder': TransformerDecoderConfig, 
        'transformer_decoder_detr': TransformerDecoderConfig, # Alias
        'rnn': RNNConfig,
        'bilstm': BiLSTMConfig
    }
    # Special handling for Transformer configs which might have extra HookedTransformer args
    # For now, let's use the helper but note that passing extra args to HookedTransformerConfig might be tricky if not filtered
    # Our helper filters, so keys missing from our subclass definition but present in parent might be lost if we don't be careful.
    # Actually TransformerConfig inherits HookedTransformerConfig. Dataclass fields should include parent fields.
    sequence_model_config = get_config_obj(seq_type, mc_data.get('sequence_model_config'), seq_cls_map, TransformerEncoderConfig)

    # Heads
    head_type = mc_data.get('head_type')
    head_cls_map = {
        'linear': LinearHeadConfig, # and 'ctc' shares this
        'ctc': LinearHeadConfig,
        'mlp': MLPHeadConfig,
        'classification': ClassificationHeadConfig
    }
    head_config = get_config_obj(head_type, mc_data.get('head_config'), head_cls_map, LinearHeadConfig)
    
    model_config = ModelConfig(
        encoder_type=encoder_type,
        encoder_config=encoder_config,
        projector_type=proj_type,
        projector_config=projector_config,
        sequence_model_type=seq_type,
        sequence_model_config=sequence_model_config,
        head_type=head_type,
        head_config=head_config,
        d_model=mc_data.get('d_model', 256),
        d_vocab=mc_data.get('d_vocab', 62),
        loss_type=mc_data.get('loss_type', 'ctc')
    )
    
    return ExperimentConfig(
        experiment_name=data.get('experiment_name', 'custom_run'),
        dataset_config=dataset_config,
        training_config=training_config,
        model_config=model_config,
        metadata_path=data.get('metadata_path', 'data/metadata.json'),
        image_base_dir=data.get('image_base_dir', 'data/images')
    )

def load_config(config_path: str) -> ExperimentConfig:
    """Load config from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    return hydrate_config(data)

def main():
    parser = argparse.ArgumentParser(description="Captcha Experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- PROVISION COMMAND ---
    # To easily create a default config file? Maybe later.
    
    # --- GENERATE COMMAND ---
    gen_parser = subparsers.add_parser("generate", help="Generate dataset")
    gen_parser.add_argument("--config-file", type=str, required=True, help="Path to dataset config file (YAML)")
    gen_parser.add_argument("--word-file", type=str, default=None, help="Path to words file (TSV) (overrides config)")
    gen_parser.add_argument("--font-root", type=str, default="fonts", help="Root of font directory")
    gen_parser.add_argument("--out-dir", type=str, default="dataset", help="Output directory")
    gen_parser.add_argument("--count", type=int, default=None, help="Limit number of words (optional)")
    
    # --- TRAIN COMMAND ---
    train_parser = subparsers.add_parser("train", help="Train model")
    # Training overrides are still useful for quick experiments without changing yaml
    train_parser.add_argument("--config-file", type=str, required=True, help="Path to experiment config file (YAML)")
    train_parser.add_argument("--metadata-path", type=str, default=None, help="Override metadata path")
    train_parser.add_argument("--image-base-dir", type=str, default=None, help="Override image base dir")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train_parser.add_argument("--wandb-project", type=str, default=None, help="WandB Project Name")

    args = parser.parse_args()
    
    # Load Config Logic
    try:
        if args.command == "generate":
            print(f"Loading dataset config from {args.config_file}...")
            if not Path(args.config_file).exists():
                raise FileNotFoundError(f"Config file not found: {args.config_file}")
            
            with open(args.config_file, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            # Support loading nested 'dataset_config' key or root keys
            if 'dataset_config' in data:
                ds_data = data['dataset_config']
            else:
                ds_data = data
                
            # Filter keys valid for DatasetConfig
            valid_keys = DatasetConfig.__dataclass_fields__.keys()
            filtered_args = {k: v for k, v in ds_data.items() if k in valid_keys}
            dataset_config = DatasetConfig(**filtered_args)
            
        elif args.command == "train":
            print(f"Loading experiment config from {args.config_file}...")
            config = load_config(args.config_file)
            
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    if args.command == "generate":
        print("Prepare generation...")
        
        # We have dataset_config from above block logic locally in main scope?
        # Re-organize main to separate command logic better or access it from scope.
        # Let's access 'dataset_config' variable safely.
        # To reuse the variable, we need to ensure it's available.
        
        if dataset_config.fonts:
            print(f"Using {len(dataset_config.fonts)} fonts from config.")
        else:
            # --- Font Selection Logic ---
            all_fonts = get_ttf_files(args.font_root)
            if not all_fonts:
                print(f"No fonts found in {args.font_root}")
                sys.exit(1)
                
            from collections import defaultdict
            family_seen = defaultdict(int)
            selected_fonts = []
            max_fonts = dataset_config.max_fonts_per_family
            
            for p in all_fonts:
                family = Path(p).parent.name
                if family_seen[family] < max_fonts:
                    selected_fonts.append(p)
                    family_seen[family] += 1
            
            # Update config with runtime paths
            dataset_config.fonts = selected_fonts

        if not dataset_config.fonts:
             print("Error: No fonts selected.")
             sys.exit(1)
             
        print(f"Selected {len(dataset_config.fonts)} fonts.")
        
        if args.word_file:
            dataset_config.word_path = args.word_file
            
        if not dataset_config.word_path:
             print("Error: --word-file not specified and 'word_path' not in config.")
             sys.exit(1)
        
        words = get_words(dataset_config.word_path)
        if args.count:
            words = words[:args.count]
            
        print(f"Generating {len(words)} samples to {args.out_dir} from {dataset_config.word_path}...")
        
        # Word transform
        transform = random_capitalize if dataset_config.random_capitalize else None
        
        generator = CaptchaGenerator(
            config=dataset_config,
            out_dir=args.out_dir,
            metadata_path=Path(args.out_dir) / "metadata.json",
            word_transform=transform
        )
        
        generator.generate(words)
        
    elif args.command == "train":
        print("Prepare training...")
        
        # Overrides
        if args.metadata_path:
            config.metadata_path = args.metadata_path
        if args.image_base_dir:
            config.image_base_dir = args.image_base_dir
        if args.epochs:
            config.training_config.epochs = args.epochs
        if args.batch_size:
            config.training_config.batch_size = args.batch_size
            
        train(config)

if __name__ == "__main__":
    main()
