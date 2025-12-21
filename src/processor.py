
import json
import torch
from torchvision import transforms
from PIL import Image
from typing import List, Union, Optional
from captcha_ocr.config.config import ExperimentConfig
from captcha_ocr.decoding import decode_ctc, decode_simple

class CaptchaProcessor:
    """
    Unified Processor for Captcha Models.
    Behaves differently based on config.
    
    1. 'cnn-transformer-detr': 
       - Height 70
       - Width formula: 28k + 14 (for Unfold patches)
       - Standard decoding (Argmax -> Char)
       
    2. 'asymmetric-convnext-transformer':
       - Height 80
       - Width formula: Multiple of 4 (Stem Stride)
       - CTC decoding (Collapse repeats, remove blanks)
    """
    def __init__(self, config: ExperimentConfig, metadata_path: str = None, vocab: List[str] = None):
        self.config = config
        # Use simple fallback if d_vocab/n_ctx not deeply configured
        self.max_seq_len = config.model_config.sequence_model_config.n_ctx

        self.target_height = config.dataset_config.height
        self.width_divisor = config.dataset_config.width_divisor
        self.width_bias = config.dataset_config.width_bias
        
        # Determine decoding strategy
        self.decoding_type = 'ctc' if config.model_config.head_type == 'ctc' else 'simple'
            
        # --- Vocab Setup ---
        if vocab is not None:
            self.chars = sorted(list(set(vocab)))
        elif metadata_path is not None:
            self.chars = self.build_vocab_from_metadata(metadata_path)
        else:
            self.chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # 0 is reserved.
        # For DETR: usually PAD/Unknown.
        # For CTC: 0 is strictly the BLANK token.
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.chars)}
        self.char_to_idx["<PAD>"] = 0 # Acts as Blank for CTC, Pad for Detr
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def build_vocab_from_metadata(metadata_path: str) -> List[str]:
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            unique_chars = set()
            for entry in data:
                text = entry.get('word_rendered', '')
                unique_chars.update(list(text))
            return sorted(list(unique_chars))
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # --- Resizing Logic ---
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Unified resizing logic based on config."""
        w, h = image.size
        scale = self.target_height / h
        new_w = int(w * scale)
        
        # Align width to divisor
        if self.width_divisor > 1:
            # For legacy DETR (div 28), the formula was specific (28k + 14).
            # For general use, rounding to nearest divisor is usually sufficient.
            # If strict legacy compatibility is needed, we can check specific flags.
            # Formula: width = divisor * k + bias
            # k = round((width - bias) / divisor)
            k = round((new_w - self.width_bias) / self.width_divisor)
            target_w = k * self.width_divisor + self.width_bias
            
            # Ensure at least k=1 if that makes sense, or just ensure > 0
            # Ideally target_w should be at least divisor + bias
            min_w = self.width_divisor + self.width_bias
            target_w = max(min_w, target_w)
        else:
            target_w = new_w
            
        return image.resize((target_w, self.target_height), resample=Image.Resampling.LANCZOS)

    def process_image(self, image: Image.Image) -> torch.Tensor:
        image = self._resize_image(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.to_tensor(image)

    # --- Text Encoding ---

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = [self.char_to_idx.get(c, 0) for c in text]
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

    # --- Decoding Logic ---

    def decode(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if self.decoding_type == 'ctc':
            return decode_ctc(token_ids, self.idx_to_char, blank_idx=0)
        else:
            return decode_simple(token_ids, self.idx_to_char)

    def __call__(self, image_path: str, text: str = None):
        try:
            image = Image.open(image_path)
            pixel_values = self.process_image(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        result = {"pixel_values": pixel_values}
        
        if text is not None:
            input_ids = self.encode_text(text)
            result["input_ids"] = input_ids
            # For CTC, it is often useful to know the target length for loss calculation
            result["target_length"] = len(text) 
            
        return result