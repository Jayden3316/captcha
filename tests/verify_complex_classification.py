
import sys
import os
import torch
from pathlib import Path

# Add project root to path
# Use absolute path of current file's parent's parent
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project path added: {project_root}")
import src.config.config as config_module
print("src.config.config imported.")

if 'SequencePoolingAdapterConfig' not in dir(config_module):
    print("ERROR: SequencePoolingAdapterConfig NOT found in src.config.config")
    print("Available:", dir(config_module))
    sys.exit(1)

from src.config.config import (
    ModelConfig, TaskType, PipelineType, ResNetEncoderConfig, 
    VerticalFeatureAdapterConfig, SequencePoolingAdapterConfig,
    TransformerEncoderConfig, ClassificationHeadConfig
)
from src.architecture.model import CaptchaModel

def verify_complex_classification():
    print("Verifying EncoderSequenceModelForClassification...")
    
    config = ModelConfig(
        task_type=TaskType.CLASSIFICATION,
        pipeline_type=PipelineType.SEQUENCE_CLASSIFICATION, # Explicit selection
        d_vocab=10, # Match head config
        encoder_type='resnet',
        encoder_config=ResNetEncoderConfig(dims=[64, 128, 256, 512]),
        
        # Enable complex pipeline
        encoder_adapter_type='vertical_feature',
        encoder_adapter_config=VerticalFeatureAdapterConfig(output_dim=1024),
        
        sequence_model_type='transformer_encoder',
        sequence_model_config=TransformerEncoderConfig(d_model=1024, n_layers=2, n_heads=4, d_vocab=10),
        
        sequence_adapter_type='sequence_pool',
        sequence_adapter_config=SequencePoolingAdapterConfig(pool_type='mean'),
        
        head_type='classification',
        head_config=ClassificationHeadConfig(d_model=1024, num_classes=10)
    )
    
    # Instantiate
    model = CaptchaModel(config)
    print("Model instantiated successfully.")
    
    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, 80, 200)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    print("Verification PASSED.")

if __name__ == "__main__":
    verify_complex_classification()
