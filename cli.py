import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional

from src.config.config import (
    ExperimentConfig, DatasetConfig, TrainingConfig, ModelConfig,
    AsymmetricConvNextEncoderConfig, ResNetEncoderConfig, 
    VerticalFeatureAdapterConfig, GlobalPoolingAdapterConfig, FlattenAdapterConfig,
    LinearProjectorConfig, MLPProjectorConfig, IdentityProjectorConfig, BottleneckProjectorConfig, ResidualProjectorConfig,
    TransformerEncoderConfig, TransformerDecoderConfig, RNNConfig, BiLSTMConfig,
    LinearHeadConfig, MLPHeadConfig, ClassificationHeadConfig
)
from src.train import train

from generate_captchas import get_ttf_files, get_words, CaptchaGenerator, random_capitalize

from src.config.loader import hydrate_config


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
    gen_parser.add_argument("--dataset-count", type=int, default=None, help="Target number of captchas to generate")
    
    # --- TRAIN COMMAND ---
    train_parser = subparsers.add_parser("train", help="Train model")
    # Training overrides are still useful for quick experiments without changing yaml
    train_parser.add_argument("--config-file", type=str, required=True, help="Path to experiment config file (YAML)")
    train_parser.add_argument("--metadata-path", type=str, default=None, help="Override metadata path")
    train_parser.add_argument("--train-metadata-path", type=str, default=None, help="Explicit training metadata path")
    train_parser.add_argument("--val-metadata-path", type=str, default=None, help="Explicit validation metadata path")
    train_parser.add_argument("--image-base-dir", type=str, default=None, help="Override image base dir")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train_parser.add_argument("--wandb-project", type=str, default=None, help="WandB Project Name")

    # --- EVALUATE COMMAND ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    eval_parser.add_argument("--metadata-path", type=str, required=True, help="Path to evaluation metadata")
    
    # --- INFERENCE COMMAND ---
    inf_parser = subparsers.add_parser("inference", help="Run inference")
    inf_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    inf_parser.add_argument("--image-paths", type=str, nargs='+', help="List of image paths")
    inf_parser.add_argument("--image-dir", type=str, help="Directory of images")

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
        if args.dataset_count:
            import random
            
            # Validation for expanding dataset beyond vocabulary size
            if args.dataset_count > len(words):
                # Check for variation factors (multiple fonts or random capitalization)
                has_variation = (len(dataset_config.fonts) > 1) or dataset_config.random_capitalize
                
                if not has_variation:
                    print(f"Error: Requested dataset_count ({args.dataset_count}) > vocabulary size ({len(words)})")
                    print("but no variation enabled (single font and no random capitalization).")
                    print("Enable random_capitalize or provide multiple fonts.")
                    sys.exit(1)
                
                # Resample with replacement to reach target count
                print(f"Expanding vocabulary from {len(words)} to {args.dataset_count} using random sampling...")
                words = random.choices(words, k=args.dataset_count)
            else:
                # Subsample without replacement (or just slice, but sample is better for diversity if list is sorted)
                # However, original behavior was slice. Let's strictly follow request: "dataset count"
                # If we want a specific size subset, random sample is usually preferred to avoid bias if list is sorted.
                words = random.sample(words, k=args.dataset_count) # Random subset

            
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
        if args.train_metadata_path:
            config.train_metadata_path = args.train_metadata_path
        if args.val_metadata_path:
            config.val_metadata_path = args.val_metadata_path
        if args.image_base_dir:
            config.image_base_dir = args.image_base_dir
        if args.epochs:
            config.training_config.epochs = args.epochs
        if args.batch_size:
            config.training_config.batch_size = args.batch_size
            
        train(config)
        
    elif args.command == "evaluate":
        print("Prepare evaluation...")
        from src.evaluate import evaluate
        
        evaluate(
            checkpoint_path=args.checkpoint,
            metadata_path=args.metadata_path
        )
        
    elif args.command == "inference":
        print("Prepare inference...")
        from src.inference import run_inference
        
        # Resolve image paths
        image_paths = []
        if args.image_paths:
            image_paths.extend(args.image_paths)
            
        if args.image_dir:
            import os
            d = Path(args.image_dir)
            if d.exists() and d.is_dir():
                # naive glob
                exts = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
                for ext in exts:
                    image_paths.extend([str(p) for p in d.rglob(ext)])
            else:
                print(f"Error: Directory not found {args.image_dir}")
                sys.exit(1)
                
        if not image_paths:
            print("Error: No images provided via --image-paths or --image-dir")
            sys.exit(1)
            
        run_inference(
            checkpoint_path=args.checkpoint,
            image_paths=image_paths
        )

if __name__ == "__main__":
    main()
