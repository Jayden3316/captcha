
import sys
import os
import torch

# Add project root to path
sys.path.append(os.getcwd())

from src.config.config import ConvNextEncoderConfig
from src.architecture.components.encoders import ConvNextEncoder

def test_zero_block_count():
    print("Testing ConvNextEncoder with 0 blocks in last stage...")
    
    # [2, 3, 2, 0]
    cfg = ConvNextEncoderConfig(
        stage_block_counts=[2, 3, 2, 0],
        dims=[64, 128, 256, 512]
    )
    
    model = ConvNextEncoder(cfg)
    
    print(f"Stage 1 blocks: {len(model.stage1)}")
    print(f"Stage 2 blocks: {len(model.stage2)}")
    print(f"Stage 3 blocks: {len(model.stage3)}")
    print(f"Stage 4 blocks: {len(model.stage4)}")
    
    assert len(model.stage4) == 0
    
    x = torch.randn(1, 3, 80, 200)
    out = model(x)
    print(f"Forward pass successful. Output shape: {out.shape}")

if __name__ == "__main__":
    test_zero_block_count()
