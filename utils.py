import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time

import pytesseract
from PIL import Image

import torch
import torchvision.transforms as transforms
import numpy as np 

def extract_text_tesseract(
    image_path: str,
    config: str = '--psm 8',
    target_width: int = None,
    target_height: int = None,
    upsample: bool = False
) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image_path: Path to the image file
        config: Tesseract configuration string
               --psm 7: Treat the image as a single text line
        target_width: Target width for upsampling (maintains aspect ratio)
        target_height: Target height for upsampling (maintains aspect ratio)
        upsample: Whether to upsample the image before OCR
    
    Returns:
        Extracted text string
    """
    try:
        img = Image.open(image_path)
        
        # Upsample if requested
        if upsample:
            if target_width is None and target_height is None:
                raise ValueError("target_width or target_height must be provided when upsample=True")
            img = upsample_image(img, target_width=target_width, target_height=target_height)
        
        text = pytesseract.image_to_string(img, config=config)
        # Strip whitespace and newlines
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""


def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_metrics(ground_truth: str, prediction: str) -> Dict:
    """
    Calculate various accuracy metrics.
    
    Returns:
        Dictionary with exact_match, case_insensitive_match, edit_distance, 
        character_accuracy, and word_correct (boolean)
    """
    exact_match = ground_truth == prediction
    case_insensitive_match = ground_truth.lower() == prediction.lower()
    edit_distance = calculate_edit_distance(ground_truth, prediction)
    
    # Character-level accuracy
    max_len = max(len(ground_truth), len(prediction))
    if max_len > 0:
        char_accuracy = 1.0 - (edit_distance / max_len)
    else:
        char_accuracy = 1.0
    
    return {
        'exact_match': exact_match,
        'case_insensitive_match': case_insensitive_match,
        'edit_distance': edit_distance,
        'character_accuracy': char_accuracy,
        'word_correct': exact_match
    }

def upsample_image(
    img: Image.Image,
    target_width: int = None,
    target_height: int = None,
    resample: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Upsample an image while maintaining aspect ratio.
    
    Args:
        img: PIL Image object
        target_width: Target width in pixels (None to ignore)
        target_height: Target height in pixels (None to ignore)
        resample: Resampling filter (default: LANCZOS for high quality)
    
    Returns:
        Upsampled PIL Image
    
    Raises:
        ValueError: If neither target_width nor target_height is provided
    """
    if target_width is None and target_height is None:
        raise ValueError("At least one of target_width or target_height must be provided")
    
    original_width, original_height = img.size
    
    # Calculate scaling factor based on provided dimensions
    if target_width is not None and target_height is not None:
        # Both provided: use the dimension that requires less upscaling
        width_scale = target_width / original_width
        height_scale = target_height / original_height
        scale = min(width_scale, height_scale)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    elif target_width is not None:
        # Only width provided: scale based on width
        scale = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale)
    else:
        # Only height provided: scale based on height
        scale = target_height / original_height
        new_width = int(original_width * scale)
        new_height = target_height
    
    # Only upsample if the target is larger than original
    if new_width > original_width or new_height > original_height:
        return img.resize((new_width, new_height), resample=resample)
    else:
        return img

class CaptchaProcessor:
    """
    Processor for preparing data for the CaptchaModel.
    Handles image resizing/normalization and text tokenization.
    """
    def __init__(self, metadata_path: str = None, vocab: List[str] = None):
        self.height = 70
        self.max_seq_len = 128
        
        # Build vocabulary
        if vocab is not None:
            self.chars = sorted(list(set(vocab)))
        elif metadata_path is not None:
            self.chars = self.build_vocab_from_metadata(metadata_path)
        else:
            # Fallback default vocabulary
            self.chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # 0 is reserved for PAD/Unknown
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.chars)}
        self.char_to_idx["<PAD>"] = 0
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        # Image transforms
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def build_vocab_from_metadata(metadata_path: str) -> List[str]:
        """
        Extract unique characters from the metadata JSON file to build the vocabulary.
        """
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            unique_chars = set()
            for entry in data:
                # Use 'word_rendered' as the ground truth text
                text = entry.get('word_rendered', '')
                unique_chars.update(list(text))
            
            return sorted(list(unique_chars))
        except Exception as e:
            print(f"Error reading metadata from {metadata_path}: {e}")
            return list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Resize image to height 70 and width nearest multiple of 35 that is greater than 70.
        Convert to tensor [C, H, W] where C=3 (RGB).
        """
        # Resize maintaining aspect ratio to height 70
        w, h = image.size
        scale = self.height / h
        new_w = int(w * scale)
        image = image.resize((new_w, self.height), resample=Image.Resampling.LANCZOS)
        
        k = round((new_w - 14) / 28)
        k = max(1, k) # Ensure at least 1 token (width 42)
        
        target_w = 28 * k + 14
        
        if target_w != new_w:
            image = image.resize((target_w, self.height), resample=Image.Resampling.LANCZOS)
        
        # Convert to RGB (3 channels)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        tensor = self.to_tensor(image) # [3, 70, W]
        return tensor

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Convert text string to token IDs and pad to max_seq_len.
        """
        # Use 0 (PAD) for unknown characters
        tokens = [self.char_to_idx.get(c, 0) for c in text] 
        
        # Pad to max_seq_len
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
            
        return torch.tensor(tokens, dtype=torch.long)

    def decode_text(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        """
        Convert token IDs back to string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        text = []
        for idx in token_ids:
            if idx == 0: # PAD
                continue
            text.append(self.idx_to_char.get(idx, ""))
        return "".join(text)

    def __call__(self, image_path: str, text: str = None):
        """
        Process a single sample.
        """
        try:
            image = Image.open(image_path)
            pixel_values = self.process_image(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        if text is not None:
            input_ids = self.encode_text(text)
            return {"pixel_values": pixel_values, "input_ids": input_ids}
        
        return {"pixel_values": pixel_values}