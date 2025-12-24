
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("."))

from src.config.config import ExperimentConfig, DatasetConfig, TrainingConfig, ModelConfig, VerticalFeatureAdapterConfig
from src.train import train, OnTheFlyDataset
from src.processor import CaptchaProcessor


import glob

def test_onthefly_config():
    print("Testing Config Hydration...")
    
    # Find some fonts
    fonts = glob.glob(os.path.abspath("val_font_library/**/*.ttf"), recursive=True)
    if not fonts:
        fonts = glob.glob(os.path.abspath("train_font_library/**/*.ttf"), recursive=True)
    
    if not fonts:
        print("WARNING: No fonts found for testing! Generation will likely fail.")
        fonts = []
    else:
        print(f"Found {len(fonts)} fonts, using first one: {fonts[0]}")
        
    # Mock config
    config = ExperimentConfig(
        experiment_name="test_onthefly",
        dataset_config=DatasetConfig(
            vocab="ABC",
            min_word_len=2,
            max_word_len=4,
            width=100,
            target_height=40,
            fonts=[fonts[0]] if fonts else []
        ),
        training_config=TrainingConfig(
            training_steps=10,
            use_onthefly_generation=True,
            batch_size=2,
            val_check_interval_steps=5,
            save_every_steps=10,
            val_steps=2,
            epochs=1, # Should be ignored conceptually
            device="cpu"
        ),
        model_config=ModelConfig(
            d_model=32,
            encoder_type="convnext",
            sequence_model_type="rnn",
            head_type="ctc",
            adapter_type="vertical_feature",
            adapter_config=VerticalFeatureAdapterConfig(output_dim=1024)
        )
    )
    
    print("Config created successfully.")
    return config

def test_dataset_generation(config):
    print("Testing OnTheFlyDataset Generation...")
    processor = CaptchaProcessor(config=config, metadata_path=None, vocab=list(config.dataset_config.vocab))
    ds = OnTheFlyDataset(config, processor, is_validation=False)
    
    print(f"Dataset Length: {len(ds)}")
    
    item = ds[0]
    print("Generated Item Keys:", item.keys())
    print("Pixel Values Shape:", item["pixel_values"].shape)
    print("Input IDs:", item["input_ids"])
    print("Gen Time:", item["gen_time"])
    
    assert item["pixel_values"].ndim == 3
    assert len(item["input_ids"]) > 0
    print("Dataset test passed.")

def test_training_loop(config):
    print("Testing Training Loop Execution...")
    # Reduce steps for speed
    config.training_config.training_steps = 4
    config.training_config.val_check_interval_steps = 2
    config.training_config.save_every_steps = 4
    config.training_config.metrics = [] # No metrics for speed
    
    # We need to mock wandb to avoid login prompts or errors, or just let it init in offline mode?
    # Or just rely on dryrun? We can use os.environ
    os.environ["WANDB_MODE"] = "disabled"
    
    # Run train
    # It might create directories
    try:
        train(config)
        print("Training loop finished successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    config = test_onthefly_config()
    test_dataset_generation(config)
    test_training_loop(config)
