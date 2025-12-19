import json
import torch
from torchvision import transforms
from PIL import Image
from typing import List, Union, Optional
from config import CaptchaConfig

class CaptchaProcessor:
    """
    Unified Processor for Captcha Models.
    Behaves differently based on config.model_type:
    
    1. 'cnn-transformer-detr': 
       - Height 70
       - Width formula: 28k + 14 (for Unfold patches)
       - Standard decoding (Argmax -> Char)
       
    2. 'asymmetric-convnext-transformer':
       - Height 80
       - Width formula: Multiple of 4 (Stem Stride)
       - CTC decoding (Collapse repeats, remove blanks)
    """
    def __init__(self, config: CaptchaConfig, metadata_path: str = None, vocab: List[str] = None):
        self.config = config
        self.model_type = config.model_type
        self.max_seq_len = config.n_ctx
        
        # --- Architecture Configuration ---
        if self.model_type == 'cnn-transformer-detr':
            self.target_height = 70
        elif self.model_type == 'asymmetric-convnext-transformer':
            self.target_height = 80
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
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

    def _resize_detr_style(self, image: Image.Image) -> Image.Image:
        """Legacy resize for 'cnn-transformer-detr' (Height 70, Unfold compatible)."""
        w, h = image.size
        scale = self.target_height / h
        new_w = int(w * scale)
        
        # Formula: width must be 28*k + 14
        k = round((new_w - 14) / 28)
        k = max(1, k)
        target_w = 28 * k + 14
        
        return image.resize((target_w, self.target_height), resample=Image.Resampling.LANCZOS)

    def _resize_convnext_style(self, image: Image.Image) -> Image.Image:
        """New resize for 'asymmetric-convnext-transformer' (Height 80, Stride 4)."""
        w, h = image.size
        scale = self.target_height / h
        new_w = int(w * scale)
        
        # Formula: width must be divisible by 4
        target_w = round(new_w / 4) * 4
        target_w = max(4, target_w)
        
        return image.resize((target_w, self.target_height), resample=Image.Resampling.LANCZOS)

    def process_image(self, image: Image.Image) -> torch.Tensor:
        # Dispatch based on flag
        if self.model_type == 'cnn-transformer-detr':
            image = self._resize_detr_style(image)
        else:
            image = self._resize_convnext_style(image)
        
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

    def _decode_ctc(self, token_ids: List[int]) -> str:
        """Greedy CTC Decoding: Collapse repeats, remove blanks (0)."""
        text = []
        prev_token = -1
        for token in token_ids:
            if token == 0: # Blank
                prev_token = token
                continue
            if token == prev_token: # Repeat
                continue
            text.append(self.idx_to_char.get(token, ""))
            prev_token = token
        return "".join(text)

    def _decode_simple(self, token_ids: List[int]) -> str:
        """Standard Decoding: Just map every token to char (ignore 0/Pad)."""
        text = []
        for token in token_ids:
            if token == 0: # Pad
                continue
            text.append(self.idx_to_char.get(token, ""))
        return "".join(text)

    def decode(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        """Public decode method that dispatches based on model type."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if self.model_type == 'asymmetric-convnext-transformer':
            return self._decode_ctc(token_ids)
        else:
            return self._decode_simple(token_ids)

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