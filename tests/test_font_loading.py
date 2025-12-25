import unittest
from unittest.mock import patch, MagicMock
from src.config.config import ExperimentConfig, DatasetConfig, ModelConfig
from src.train import OnTheFlyDataset

class TestFontLoading(unittest.TestCase):
    def setUp(self):
        # Dummy processor
        self.processor = MagicMock()
        self.processor.chars = list("ABC")
        self.processor.char_to_idx = {"A": 1, "B": 2, "C": 3, "<PAD>": 0}
        self.processor.idx_to_char = {1: "A", 2: "B", 3: "C", 0: "<PAD>"}

        self.dataset_config = DatasetConfig(
            width=100, height=32,
            vocab="ABC"
        )
        self.config = ExperimentConfig(
            experiment_name="test_font",
            dataset_config=self.dataset_config,
            model_config=ModelConfig(adapter_type="flatten")
        )

    @patch('src.utils.get_ttf_files')
    @patch('src.train.ConfigurableImageCaptcha')
    def test_font_root_loading(self, mock_captcha_cls, mock_get_ttf):
        """Test loading fonts from font_root"""
        self.dataset_config.font_root = "dummy_root"
        self.dataset_config.fonts = None
        
        mock_get_ttf.return_value = ["font1.ttf", "font2.ttf"]
        
        dataset = OnTheFlyDataset(self.config, self.processor, is_validation=False)
        
        mock_get_ttf.assert_called_with("dummy_root")
        # Check that fonts passed to generator are correct via filtered kwargs trick or direct check if we passed direct args
        # We passed explicit args now
        args, kwargs = mock_captcha_cls.call_args
        # In our implementation we pass named args.
        self.assertEqual(kwargs['fonts'], ["font1.ttf", "font2.ttf"])

    @patch('src.utils.get_ttf_files')
    @patch('src.train.ConfigurableImageCaptcha')
    def test_train_val_split(self, mock_captcha_cls, mock_get_ttf):
        """Test separate roots for train/val"""
        self.dataset_config.train_font_root = "train_root"
        self.dataset_config.val_font_root = "val_root"
        self.dataset_config.fonts = None
        
        mock_get_ttf.side_effect = lambda root: [f"{root}_font.ttf"]
        
        # Train dataset
        ds_train = OnTheFlyDataset(self.config, self.processor, is_validation=False)
        args_train, kwargs_train = mock_captcha_cls.call_args_list[-1] # call_args gives LAST call
        self.assertEqual(kwargs_train['fonts'], ["train_root_font.ttf"])
        
        # Val dataset
        ds_val = OnTheFlyDataset(self.config, self.processor, is_validation=True)
        args_val, kwargs_val = mock_captcha_cls.call_args # Still the last call? No, ds_val was last
        self.assertEqual(kwargs_val['fonts'], ["val_root_font.ttf"])

    @patch('src.train.ConfigurableImageCaptcha')
    def test_explicit_fonts_priority(self, mock_captcha_cls):
        """Test that explicit fonts list overrides root"""
        self.dataset_config.fonts = ["explicit.ttf"]
        self.dataset_config.font_root = "ignored_root"
        
        dataset = OnTheFlyDataset(self.config, self.processor)
        
        args, kwargs = mock_captcha_cls.call_args
        self.assertEqual(kwargs['fonts'], ["explicit.ttf"])

if __name__ == '__main__':
    unittest.main()
