
import unittest
import torch
import yaml
import os
from src.architecture.model import CaptchaModel
from src.config.loader import hydrate_config

class TestConfiguration(unittest.TestCase):
    def test_asymm_x_resnet_base_config_loading(self):
        # 1. Load the YAML configuration
        config_path = "experiments/training_configs/classification/asymm_x_resnet_base.yaml"
        # Ensure we are running from the root or adjust path
        if not os.path.exists(config_path):
             # Try absolute path if relative fails (assuming running from root)
             # User gave absolute path in request: c:\Users\jerry\IITM\CFI\Ai Club\2025-26\Precog\captcha_ocr\...
             # But tests are usually run from root or tests dir. 
             # Let's assume root for now given the @ syntax typically implies repo root relative.
             pass
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 2. Hydrate configuration
        experiment_config = hydrate_config(config_dict)
        model_config = experiment_config.model_config
        
        # 3. Instantiate Model
        model = CaptchaModel(model_config)
        
        # 4. Generate dummy input
        # Dataset config specifies width 192, height 64
        # ResNet expects [B, C, H, W]
        batch_size = 2
        channels = 3
        height = experiment_config.dataset_config.height
        width = experiment_config.dataset_config.width
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        
        # 5. Run forward pass
        print(f"Testing Model Forward Pass with Input: {dummy_input.shape}")
        output = model(dummy_input)
        
        # 6. Verify Output
        print(f"Model Output Shape: {output.shape}")
        
        # Check output shape: [Batch, NumClasses]
        # num_classes is in head_config
        expected_classes = model_config.head_config.num_classes
        
        self.assertEqual(output.shape, (batch_size, expected_classes))
        
if __name__ == '__main__':
    unittest.main()
