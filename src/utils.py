import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import re

# Lazy import sklearn to avoid overhead if not used? 
# No, standard import is fine.
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
    Calculate various accuracy metrics (Legacy/OCR focus).
    
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

def calculate_classification_metrics(targets: List[Union[str, int]], preds: List[Union[str, int]], metrics_list: List[str]) -> Dict[str, float]:
    """
    Calculate classification metrics using sklearn.
    
    Args:
        targets: List of ground truth values (strings or ints)
        preds: List of predicted values (strings or ints)
        metrics_list: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    # Common classification metrics
    if 'accuracy' in metrics_list:
        results['accuracy'] = accuracy_score(targets, preds)
        
    if 'f1' in metrics_list:
        results['f1'] = f1_score(targets, preds, average='macro', zero_division=0)
        
    if 'precision' in metrics_list:
        results['precision'] = precision_score(targets, preds, average='macro', zero_division=0)
        
    if 'recall' in metrics_list:
        results['recall'] = recall_score(targets, preds, average='macro', zero_division=0)
        
    return results

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


def sanitize_alnum(text: str) -> str:
    """
    Keep only alphanumeric characters (A–Z, a–z, 0–9) in the string.
    """
    return re.sub(r"[^0-9A-Za-z]", "", text)


def get_words(
    file_path: str,
    min_word_len: int = 4,
    max_word_len: Optional[int] = None,
) -> List[str]:
    """
    Load words from a file (TSV, CSV, or plain text) and filter by length and alnum content.
    """
    try:
        # Try reading as TSV with expected columns
        df = pd.read_csv(file_path, sep="\t", names=["word_id", "word", "frequency"], on_bad_lines='skip')
        
        # Check if 'word' column is usable (not all NaN)
        if df['word'].isna().all():
             # Maybe it was not TSV or names mismatch?
             raise ValueError("TSV read yielded empty 'word' column")
             
        word_list = df["word"].tolist()
        
    except Exception:
        # Fallback: Read as simple lines
        try:
             with open(file_path, 'r', encoding='utf-8') as f:
                 word_list = [l.strip() for l in f.readlines()]
        except Exception as e:
             print(f"Error reading word file {file_path}: {e}")
             return []

    words: set[str] = set()
    for word in word_list:
        if not isinstance(word, str): continue
        # Split by spaces just in case lines have multiple words
        for split_word in word.split():
            clean = sanitize_alnum(split_word)
            n = len(clean)
            if n >= min_word_len and (max_word_len is None or n <= max_word_len):
                words.add(clean)
    return sorted(list(words))


def get_ttf_files(root_path: str | Path) -> List[str]:
    """
    Recursively find all .ttf files in the given directory tree.

    Args:
        root_path: Path to the root directory to search (e.g., 'font_library')

    Returns:
        A list of string paths for all .ttf files found
    """
    root = Path(root_path)
    ttf_files: List[str] = []

    for file_path in root.rglob("*.ttf"):
        ttf_files.append(str(file_path))

    return sorted(ttf_files)
