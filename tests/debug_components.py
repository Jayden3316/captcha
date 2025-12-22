
import torch
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.architecture.components.encoders import ResNetEncoder
from src.architecture.components.adapters import FlattenAdapter

def debug_resnet():
    print("Assuming ResNet Config...")
    # Mock config object if needed, ResNetEncoder just takes cfg but doesn't seem to use it deeply in __init__ 
    encoder = ResNetEncoder(cfg=None)
    
    dummy_input = torch.randn(2, 3, 80, 200) # [B, C, H, W]
    print(f"Input: {dummy_input.shape}")
    
    output = encoder(dummy_input)
    print(f"ResNet Output: {output.shape}")
    # Expected: [2, 512, 2, 6]
    
    return output

def debug_adapter(input_tensor):
    adapter = FlattenAdapter(input_channels=512, output_dim=6144)
    output = adapter(input_tensor)
    print(f"Adapter Output: {output.shape}")
    # Expected: [2, 6144]
    return output

if __name__ == "__main__":
    try:
        out = debug_resnet()
        adapt_out = debug_adapter(out)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
