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

from src.config.config import ExperimentConfig
from src.architecture.model import CaptchaModel
from src.utils import calculate_metrics
from src.processor import CaptchaProcessor
from src.losses import get_loss_function
from src.decoding import decode_simple

class CaptchaDataset(Dataset):
    def __init__(self, metadata_path: str, processor: CaptchaProcessor, base_dir: str):
        self.processor = processor
        self.base_dir = Path(base_dir)
        
        with open(metadata_path, 'r') as f:
            raw_metadata = json.load(f)
            
        # --- Pre-Filter Step ---
        self.metadata = []
        print(f"Scanning dataset at {base_dir}...")
        
        for item in raw_metadata:
            image_path = self.base_dir / item['image_path']
            
            # Check existence immediately
            if image_path.exists():
                self.metadata.append(item)
            else:
                # Try fallback name check
                filename = Path(item['image_path']).name
                alt_path = self.base_dir / filename
                if alt_path.exists():
                    item['image_path'] = filename # Update path to correct one
                    self.metadata.append(item)
                else:
                    # Just skip it
                    print(f"Skipping missing file: {item['image_path']}")
                    
        print(f"Dataset loaded. Found {len(self.metadata)} valid samples out of {len(raw_metadata)}.")
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # We know this exists because we checked in __init__
        item = self.metadata[idx]
        image_path = self.base_dir / item['image_path']
        text = item['word_rendered']
        
        processed = self.processor(str(image_path), text)
        
        # Only crash if the image is corrupt (exists but can't be opened)
        if processed is None:
             raise ValueError(f"Corrupt image file at {image_path}")
             
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
    
    if 'target_length' in batch[0]:
        target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    else:
        # Fallback if not provided (assume full padded length, though suboptimal for CTC)
        target_lengths = torch.tensor([len(item['input_ids']) for item in batch], dtype=torch.long)

    return {
        "pixel_values": padded_images,
        "input_ids": input_ids,
        "target_lengths": target_lengths
    }

def train(
    config: "ExperimentConfig",
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
):
    
    # Initialize WandB
    wandb.init(
        project=config.training_config.wandb_project, 
        name=config.training_config.wandb_run_name,
        config=config.to_dict()
    )
    
    os.makedirs(config.training_config.checkpoint_dir, exist_ok=True)
    device = torch.device(config.training_config.device if torch.cuda.is_available() else "cpu")
    
    # Data Setup
    metadata_path = config.metadata_path
    image_base_dir = config.image_base_dir
    
    if train_dataset is None or val_dataset is None:
        # Processor initialized with full ExperimentConfig
        processor = CaptchaProcessor(config=config, metadata_path=metadata_path)
        
        # Ensure vocab size matches config if we used defaults
        # config.model_config.d_vocab = processor.vocab_size - 1
        
        full_dataset = CaptchaDataset(metadata_path, processor, image_base_dir)
        train_size = int((1.0 - config.training_config.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        if train_dataset is None:
            train_dataset = train_ds
        if val_dataset is None:
            val_dataset = val_ds
    else:
        # Extract processor from dataset if possible
        if hasattr(train_dataset, 'dataset'): # Handle Subset
             # Check if dataset has processor
            if hasattr(train_dataset.dataset, 'processor'):
                processor = train_dataset.dataset.processor
            else:
                 processor = CaptchaProcessor(config=config, metadata_path=metadata_path)
        elif hasattr(train_dataset, 'processor'):
            processor = train_dataset.processor
        else:
            processor = CaptchaProcessor(config=config, metadata_path=metadata_path)

    batch_size = config.training_config.batch_size
    num_workers = config.training_config.num_workers
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=config.training_config.shuffle_train, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    print(f"Initializing CaptchaModel with config: {config.model_config.encoder_type} + {config.model_config.sequence_model_type}...")
    
    # Use the new clean architecture
    # We pass the ModelConfig directly
    model = CaptchaModel(config.model_config)
        
    model.to(device)
    
    optimizer_cls = getattr(optim, config.training_config.optimizer_type.upper(), optim.AdamW)
    if config.training_config.optimizer_type.lower() == 'adamw':
         optimizer = optim.AdamW(model.parameters(), lr=config.training_config.learning_rate, weight_decay=config.training_config.weight_decay)
    else:
         optimizer = optimizer_cls(model.parameters(), lr=config.training_config.learning_rate)

    loss_fn = get_loss_function(config.model_config)
    
    best_val_metric = 0.0
    monitor_metric = config.training_config.monitor_metric

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(config.describe())
    print("-" * 50)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Trainable Params: {trainable_params:,}")
    print("="*50 + "\n")

    epochs = config.training_config.epochs
    grad_clip_norm = config.training_config.grad_clip_norm
    log_every = config.training_config.log_every_n_steps
    save_dir = config.training_config.checkpoint_dir

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in enumerate(progress_bar):
            if batch is None: continue
            
            images = batch["pixel_values"].to(device)
            targets = batch["input_ids"].to(device) 
            target_lengths = batch['target_lengths'].to(device)
            
            logits = model(images) 

            loss = loss_fn(logits, targets, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            if step % log_every == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        if (epoch + 1) % config.training_config.val_check_interval == 0:
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
                    target_lengths = batch["target_lengths"].to(device)
                    
                    logits = model(images)
    
                    loss = loss_fn(logits, targets, target_lengths)
                        
                    val_loss += loss.item()
                    
                    # For CTC, we need to decode. 
                    # Note: UniversalCaptchaModel returns [B, S, D] now.
                    preds = logits.argmax(dim=-1) # [B, Seq_len]
                    
                    # Calculate metrics using utils.py
                    for i in range(len(targets)):
    
                        pred_str = processor.decode(preds[i])
    
                        target_ids = targets[i][:target_lengths[i]].tolist()
                        # Use simple decoding for ground truth (no CTC logic needed)
                        target_str = decode_simple(target_ids, processor.idx_to_char)
                        
                        metrics = calculate_metrics(target_str, pred_str)
                        
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
            is_best = False
            current_metric = exact_match_acc # logic for other metrics can be added
            if current_metric >= best_val_metric:
                best_val_metric = current_metric
                is_best = True
                
            fname = f"{config.experiment_name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, fname))
            
            if is_best:
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_{config.experiment_name}.pth"))
                print(f"New best model saved with {monitor_metric}: {best_val_metric:.4f}")

if __name__ == "__main__":
    train(
        metadata_path="validation_set/metadata.json",
        image_base_dir=".", 
        batch_size=32,
        epochs=50,
        model_type='asymmetric-convnext-transformer'
    )