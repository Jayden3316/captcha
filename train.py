import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

from config import CaptchaConfig
from modelling import CaptchaModel
from utils import CaptchaProcessor, calculate_metrics

def get_vocab_from_metadata(metadata_path: str) -> Tuple[List[str], int]:
    """
    Parses metadata to find the unique characters and vocabulary size.
    Returns:
        vocab: List of unique characters
        vocab_size: Length of vocab + 1 (for PAD token)
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    unique_chars = set()
    for item in metadata:
        if 'word_rendered' in item:
            unique_chars.update(list(item['word_rendered']))
            
    vocab = sorted(list(unique_chars))
    # +1 for PAD token (index 0)
    vocab_size = len(vocab) + 1
    return vocab, vocab_size

class CaptchaDataset(Dataset):
    def __init__(self, metadata_path: str, processor: CaptchaProcessor, base_dir: str):
        self.processor = processor
        self.base_dir = Path(base_dir)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.base_dir / item['image_path']
        text = item['word_rendered']
        
        if not os.path.exists(image_path):
             if idx == 0:
                 raise FileNotFoundError(f"Image not found at {image_path} (Fallback failed)")
             print(f"Warning: Image not found at {image_path}, using fallback.")
             return self.__getitem__(0)

        processed = self.processor(str(image_path), text)
        if processed is None:
            if idx == 0:
                raise ValueError(f"Failed to process image at {image_path} (Fallback failed)")
            return self.__getitem__(0)
            
        return processed

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    pixel_values = [item['pixel_values'] for item in batch]
    max_w = max(p.shape[2] for p in pixel_values)
    
    batch_size = len(batch)
    c, h = pixel_values[0].shape[:2]
    padded_images = torch.zeros(batch_size, c, h, max_w)
    
    for i, p in enumerate(pixel_values):
        w = p.shape[2]
        padded_images[i, :, :, :w] = p
        
    input_ids = torch.stack([item['input_ids'] for item in batch])
    
    return {
        "pixel_values": padded_images,
        "input_ids": input_ids
    }

# --- Training Loop ---
def train(
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    metadata_path: str = "validation_set/metadata.json",
    image_base_dir: str = ".", 
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    wandb_project: str = "captcha-ocr"
):
    # Initialize WandB
    wandb.init(project=wandb_project, config={
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr
    })
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Setup
    if train_dataset is None or val_dataset is None:
        processor = CaptchaProcessor(metadata_path=metadata_path)
        full_dataset = CaptchaDataset(metadata_path, processor, image_base_dir)
        
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        if train_dataset is None:
            train_dataset = train_ds
        if val_dataset is None:
            val_dataset = val_ds
    else:
        if hasattr(train_dataset, 'dataset'): # Handle Subset
            processor = train_dataset.dataset.processor
        elif hasattr(train_dataset, 'processor'):
            processor = train_dataset.processor
        else:
            processor = CaptchaProcessor(metadata_path=metadata_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Model Setup
    model_config = CaptchaConfig()

    vocab, vocab_size = get_vocab_from_metadata(metadata_path)
    print(f"Detected Vocab Size: {vocab_size} (including PAD)")
    
    model_config.d_vocab = processor.vocab_size
    model_config.d_vocab_out = processor.vocab_size
    
    model = CaptchaModel(model_config)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD
    
    best_val_acc = 0.0

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 50)
    print("MODEL ARCHITECTURE")
    print("-" * 50)
    print(model)
    print("="*50 + "\n")

    print(f"Starting training on {device} with {len(train_dataset)} training samples.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            if batch is None: continue
            
            images = batch["pixel_values"].to(device)
            targets = batch["input_ids"].to(device) # [B, 56]
            
            logits = model(images) # [B, 56, Vocab]
            
            loss = loss_fn(logits.reshape(-1, processor.vocab_size), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"train_loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        total_samples = 0
        
        # Metrics Accumulators
        total_edit_distance = 0.0
        total_char_accuracy = 0.0
        exact_matches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if batch is None: continue
                
                images = batch["pixel_values"].to(device)
                targets = batch["input_ids"].to(device)
                
                logits = model(images)
                
                loss = loss_fn(logits.reshape(-1, processor.vocab_size), targets.reshape(-1))
                val_loss += loss.item()
                
                preds = logits.argmax(dim=-1) # [B, 56]
                
                # Calculate metrics using utils.py
                for p_ids, t_ids in zip(preds, targets):
                    p_text = processor.decode_text(p_ids)
                    t_text = processor.decode_text(t_ids)
                    
                    metrics = calculate_metrics(t_text, p_text)
                    
                    total_edit_distance += metrics['edit_distance']
                    total_char_accuracy += metrics['character_accuracy']
                    if metrics['exact_match']:
                        exact_matches += 1
                        
                total_samples += images.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_char_acc = total_char_accuracy / total_samples if total_samples > 0 else 0.0
        avg_edit_dist = total_edit_distance / total_samples if total_samples > 0 else 0.0
        exact_match_acc = exact_matches / total_samples if total_samples > 0 else 0.0
        
        print(f"Val Loss: {avg_val_loss:.4f} | Char Acc: {avg_char_acc:.4f} | Edit Dist: {avg_edit_dist:.4f} | Exact Match: {exact_match_acc:.4f}")
        
        wandb.log({
            "val_loss": avg_val_loss,
            "val_char_acc": avg_char_acc,
            "val_edit_distance": avg_edit_dist,
            "val_exact_match": exact_match_acc,
            "epoch": epoch + 1
        })
        
        # Checkpointing
        if exact_match_acc >= best_val_acc:
            best_val_acc = exact_match_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best model saved with Exact Match: {best_val_acc:.4f}")
            
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train(
        metadata_path="validation_set/metadata.json",
        image_base_dir=".", 
        batch_size=32,
        epochs=50
    )