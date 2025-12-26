import argparse
import yaml
import torch
import os
import sys

# Add current directory to path so imports work
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.config.loader import hydrate_config
from src.architecture.model import CaptchaModel
from src.losses import get_loss_function
from src.config.config import PipelineType

def main():
    parser = argparse.ArgumentParser(description="Test model configuration and perform dummy forward pass.")
    parser.add_argument("config_path", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # 1. Load Configuration
    print(f"\n[1/5] Loading configuration from {args.config_path}...")
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)

    try:
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Hydrate configuration
        experiment_config = hydrate_config(config_dict)
        model_config = experiment_config.model_config
        dataset_config = experiment_config.dataset_config
        print(f"  Configuration loaded successfully.")
        print(f"  > Pipeline Type: {model_config.pipeline_type}")
        print(f"  > Model Arch: Encoder={model_config.encoder_type}, Adapter={model_config.adapter_type}, Head={model_config.head_type}")
        print(f"  > Loss Type: {model_config.loss_type}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. Instantiate Model
    print("\n[2/5] Instantiating model...")
    try:
        model = CaptchaModel(model_config)
        print("  Model instantiated successfully.")
        
        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  > Total Parameters: {total_params:,}")
        print(f"  > Trainable Parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"Error instantiating model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Create Dummy Inputs
    print("\n[3/5] Generating dummy inputs...")
    try:
        # Default to 3 channels if not specified (captchas are usually RGB)
        channels = 3
        # Check if encoder config specifies different input channels
        if model_config.encoder_config and hasattr(model_config.encoder_config, 'stem_in_channels'):
            channels = model_config.encoder_config.stem_in_channels
            
        height = dataset_config.height
        width = dataset_config.width
        batch_size = 2
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        print(f"  Input Shape: {dummy_input.shape}")
        
    except Exception as e:
        print(f"Error creating dummy inputs: {e}")
        sys.exit(1)

    # 4. Run Forward Pass
    print("\n[4/5] Running forward pass...")
    # Use train mode to catch potential issues with dropout or batchnorm if single batch
    model.train() 
    
    try:
        logits = model(dummy_input)
        print(f"  Forward pass successful.")
        print(f"  > Output Logits Shape: {logits.shape}")
        
        if torch.isnan(logits).any():
            print("  [WARNING] NaNs detected in logits!")
        else:
            print("  Logits look valid (no NaNs).")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Check Loss Calculation
    print("\n[5/5] Checking loss calculation...")
    try:
        loss_fn = get_loss_function(model_config)
        
        # Generate dummy targets based on pipeline/loss type
        if model_config.pipeline_type == PipelineType.STANDARD_GENERATION:
            # CTC Loss
            # Logits: [Batch, Seq, Vocab]
            # Targets: [Batch, Seq] (indices)
            seq_len = 5 # arbitrary
            vocab_size = model_config.head_config.d_vocab
            
            # Ensure logits shape is compatible
            # Logits usually [Batch, MaxSeqLen, Vocab]
            
            targets = torch.randint(1, vocab_size, (batch_size, seq_len))
            target_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            
            print(f"  Testing CTC Loss with targets shape: {targets.shape}")
            loss = loss_fn(logits, targets, target_lengths)
            
        elif model_config.pipeline_type in [PipelineType.STANDARD_CLASSIFICATION, PipelineType.SEQUENCE_CLASSIFICATION]:
            # Cross Entropy usually
            num_classes = model_config.head_config.num_classes
            targets = torch.randint(0, num_classes, (batch_size,))
            target_lengths = torch.zeros(batch_size) # Unused
            
            print(f"  Testing Classification Loss (likely CrossEntropy) with targets shape: {targets.shape}")
            loss = loss_fn(logits, targets, target_lengths)
            
        else:
            print("  Unknown pipeline type, skipping specific loss check.")
            loss = None

        if loss is not None:
            print(f"  Loss calculation successful: {loss.item()}")
            if torch.isnan(loss):
                print("  [CRITICAL] Loss is NaN!")
            else:
                print("  Loss is valid (not NaN).")
        
    except Exception as e:
        print(f"Error during loss calculation: {e}")
        print("  (Note: This might be expected if dummy target shapes don't perfectly match model output config)")
        import traceback
        traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
