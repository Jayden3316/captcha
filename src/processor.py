
import json
import torch
from torchvision import transforms
from PIL import Image
from typing import List, Union, Optional
from abc import ABC, abstractmethod

from src.config.config import ExperimentConfig, TaskType
from src.decoding import decode_ctc, decode_simple

class BaseProcessor(ABC):
    """
    Abstract Base Processor handling image preprocessing and defining text interfaces.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.target_height = config.dataset_config.height
        self.target_width = config.dataset_config.width
        self.width_divisor = config.dataset_config.width_divisor
        self.width_bias = config.dataset_config.width_bias
        self.resize_mode = config.dataset_config.resize_mode
        
        self.to_tensor = transforms.ToTensor()
        
    # --- Resizing Logic ---
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Unified resizing logic based on config."""
        if self.resize_mode == "fixed":
            # Direct resize to target dims (stretch)
            return image.resize((self.target_width, self.target_height), resample=Image.Resampling.LANCZOS)
        
        # Variable width logic (aspect ratio preserving)
        w, h = image.size
        scale = self.target_height / h
        new_w = int(w * scale)
        
        # Align width to divisor
        if self.width_divisor > 1:
            k = round((new_w - self.width_bias) / self.width_divisor)
            target_w = k * self.width_divisor + self.width_bias
            
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

    @abstractmethod
    def encode_text(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        pass
    
    @staticmethod
    def _load_vocab_from_metadata(metadata_path: str, is_classification: bool = False) -> List[str]:
        """Helper to load vocab from metadata."""
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            if is_classification:
                # For classification, vocab is the set of input words
                words = set()
                for entry in data:
                    words.add(entry.get('word_input', ''))
                return sorted(list(words))
            else:
                # For generation, vocab is the set of unique characters
                unique_chars = set()
                for entry in data:
                    text = entry.get('word_rendered', '')
                    unique_chars.update(list(text))
                return sorted(list(unique_chars))
        except Exception as e:
            print(f"Error reading metadata: {e}")
            if is_classification:
                return [] # Should probably fail harder here for classification
            return list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

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
            
            # Target length definition varies by task
            if isinstance(self, GenerationProcessor):
                result["target_length"] = len(text)
            else:
                result["target_length"] = 1 # Classification is always length 1
            
        return result


class GenerationProcessor(BaseProcessor):
    """
    Processor for Sequence Generation / OCR tasks.
    Maps text to character sequences.
    """
    def __init__(self, config: ExperimentConfig, metadata_path: str = None, vocab: List[str] = None):
        super().__init__(config)
        
        # Use simple fallback if d_vocab/n_ctx not deeply configured
        self.max_seq_len = getattr(getattr(config.model_config, 'sequence_model_config', None), 'n_ctx', 128)

        # Determine decoding strategy
        self.decoding_type = 'ctc' if config.model_config.head_type == 'ctc' else 'simple'
            
        # --- Vocab Setup ---
        if vocab is not None:
            self.chars = sorted(list(set(vocab)))
        elif metadata_path is not None:
            self.chars = self._load_vocab_from_metadata(metadata_path, is_classification=False)
        else:
            self.chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # 0 is reserved.
        # For DETR: usually PAD/Unknown.
        # For CTC: 0 is strictly the BLANK token.
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.chars)}
        self.char_to_idx["<PAD>"] = 0 
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = [self.char_to_idx.get(c, 0) for c in text]
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if self.decoding_type == 'ctc':
            return decode_ctc(token_ids, self.idx_to_char, blank_idx=0)
        else:
            return decode_simple(token_ids, self.idx_to_char)


class ClassificationProcessor(BaseProcessor):
    """
    Processor for Classification tasks.
    Maps text (labels) to a single class index.
    """
    def __init__(self, config: ExperimentConfig, metadata_path: str = None, vocab: List[str] = None):
        super().__init__(config)
        
        # --- Vocab (Classes) Setup ---
        if vocab is not None:
            self.classes = sorted(list(set(vocab)))
        elif metadata_path is not None:
            self.classes = self._load_vocab_from_metadata(metadata_path, is_classification=True)
        else:
            # Fallback for classification is tricky, maybe numbers?
            # But usually classification implies a known set.
            # We'll assume digits for a safe fallback if nothing provided.
            self.classes = [str(i) for i in range(10)]

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        self.num_classes = len(self.classes)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encodes a label string into a tensor containing the class index.
        Returns tensor of shape [1] (or just scalar wrapped) usually, 
        but to keep collate_fn happy we might want [1] to stack easily?
        
        Actually, CrossEntropyLoss expects target as [N] (for batch).
        Our collate_fn stacks input_ids.
        If we return a scalar tensor(idx), stack -> [Batch].
        If we return tensor([idx]), stack -> [Batch, 1].
        
        Standard CrossEntropy expects [Batch].
        """
        idx = self.class_to_idx.get(text, -1)
        if idx == -1:
            print(f"Warning: Label '{text}' not in vocabulary.")
            idx = 0 # Fallback
            
        # Return as 0-d tensor so stack creates 1-d vector [Batch]
        return torch.tensor(idx, dtype=torch.long) 

    def decode(self, token_ids: Union[torch.Tensor, List[int], int]) -> str:
        """
        Decodes a class index (or single-item list/tensor) to string.
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.numel() == 1:
                idx = token_ids.item()
            else:
                # If passed a distribution or unsqueezed tensor, get argmax or squeeze?
                # Assumes we passed the predicted index.
                idx = token_ids.item()
        elif isinstance(token_ids, list):
            idx = token_ids[0]
        else:
            idx = token_ids
            
        return self.idx_to_class.get(idx, "<UNK>")


def CaptchaProcessor(config: ExperimentConfig, metadata_path: str = None, vocab: List[str] = None) -> BaseProcessor:
    """
    Factory function to create the right processor.
    Keeps legacy name to avoid breaking imports (mostly).
    """
    # Check task type
    task_type = getattr(config.model_config, 'task_type', TaskType.GENERATION)
    
    if task_type == TaskType.CLASSIFICATION:
        return ClassificationProcessor(config, metadata_path, vocab)
    else:
        return GenerationProcessor(config, metadata_path, vocab)