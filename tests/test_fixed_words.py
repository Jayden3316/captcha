import unittest
import torch
from pathlib import Path
import tempfile
import os

from src.config.config import ExperimentConfig, DatasetConfig, ModelConfig
# We need to mock Processor or use real one. Real one requires config.
from src.processor import CaptchaProcessor
from src.train import OnTheFlyDataset

class TestFixedWords(unittest.TestCase):
    def setUp(self):
        # Create a dummy config
        self.dataset_config = DatasetConfig(
            width=100, height=32,
            vocab="ABC",
            min_word_len=3, max_word_len=3,
            random_capitalize=False
        )
        self.config = ExperimentConfig(
            experiment_name="test_fixed_words",
            dataset_config=self.dataset_config,
            model_config=ModelConfig(adapter_type="flatten") # Valid adapter
        )
        # Processor needs to be initialized. 
        # We can mock it or just init it with dummy metadata path (which it might complain about if not exists)
        # Or just use the one created in train.py logic?
        # Let's instantiate it with no metadata path but set chars manually as we did in train.py logic
        self.processor = CaptchaProcessor(config=self.config, metadata_path=None)
        self.processor.chars = ["A", "B", "C"]
        # Use 1-based indexing for CTC
        self.processor.char_to_idx = {"A": 1, "B": 2, "C": 3}
        self.processor.char_to_idx["<PAD>"] = 0
        self.processor.idx_to_char = {1: "A", 2: "B", 3: "C", 0: "<PAD>"}
        
    def test_fixed_words_explicit(self):
        """Test with explicit list of words in config."""
        fixed_words = ["AAA", "BBB", "CCC"]
        self.config.dataset_config.fixed_words = fixed_words
        
        dataset = OnTheFlyDataset(self.config, self.processor)
        
        # Mock generation to avoid font loading issues
        from PIL import Image
        dataset.captcha_gen.generate_image = lambda w, bg_color=None, fg_color=None: Image.new('RGB', (100, 32))
        
        # Verify word_pool is set
        self.assertEqual(dataset.word_pool, fixed_words)
        
        # Sample a few times. Note: actual image generation happens, 
        # so we rely on randomness but since pool is small we likely hit all or proper subset.
        # But we can't easily check the image content without OCR.
        # However, we can check if it runs without error.
        # And we can verify the LENGTH match conceptually if we could decode input_ids 
        # but input_ids are from processor which encodes 'word'.
        
        item = dataset[0]
        input_ids = item['input_ids']
        item = dataset[0]
        input_ids = item['input_ids']
        # Use simple decoding (map indices to chars) to verify targets, avoiding CTC collapse logic
        decoded = "".join([self.processor.idx_to_char[idx.item()] for idx in input_ids if idx.item() != 0])
        self.assertIn(decoded, fixed_words)
        
    def test_word_path(self):
        """Test loading from a word file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("word1\nword2\nword3")
            tmp_path = f.name
            
        try:
            self.config.dataset_config.word_path = tmp_path
            self.config.dataset_config.fixed_words = None # Ensure precedence check works (fixed > path)
            
            # Note: get_words sanitizes. "word1" -> "word1". 
            # Our vocab is "ABC", so "word1" characters are NOT in vocab.
            # CaptchaGenerator might fail or Processor might fail to encode?
            # Processor encodes using `chars`. "word1" has 'w','o','r','d','1'. 
            # If not in chars, it might skip or error depending on implementation.
            # Let's update vocab/processor chars for this test.
            self.config.dataset_config.vocab = "word123"
            self.config.dataset_config.vocab = "word123"
            self.processor.chars = sorted(list("word123"))
            self.processor.char_to_idx = {c: i+1 for i, c in enumerate(self.processor.chars)}
            self.processor.char_to_idx["<PAD>"] = 0
            self.processor.idx_to_char = {i+1: c for i, c in enumerate(self.processor.chars)}
            self.processor.idx_to_char[0] = "<PAD>"
            
            # Also min_len check in get_words. "word1" is len 5.
            self.config.dataset_config.min_word_len = 1
            self.config.dataset_config.max_word_len = 10
            
            dataset = OnTheFlyDataset(self.config, self.processor)
            
            # Mock generation
            from PIL import Image
            dataset.captcha_gen.generate_image = lambda w, bg_color=None, fg_color=None: Image.new('RGB', (100, 32))

            expected = ["word1", "word2", "word3"]
            self.assertEqual(sorted(dataset.word_pool), sorted(expected))
            
            item = dataset[0]
            input_ids = item['input_ids']
            decoded = "".join([self.processor.idx_to_char[idx.item()] for idx in input_ids if idx.item() != 0])
            print(f"DEBUG: Dataset Word Pool: {dataset.word_pool}")
            print(f"DEBUG: Input IDs: {input_ids.tolist()}")
            print(f"DEBUG: Decoded: {decoded}")
            self.assertIn(decoded, expected)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
