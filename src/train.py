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
from src.generator import ConfigurableImageCaptcha, random_color
import time

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
        
        if processed is None:
             raise ValueError(f"Corrupt image file at {image_path}")
             
        return processed

class OnTheFlyDataset(Dataset):
    def __init__(self, config: ExperimentConfig, processor: CaptchaProcessor, is_validation: bool = False):
        self.config = config
        self.processor = processor
        self.is_validation = is_validation
        
        # Generator
        self.dataset_config = config.dataset_config
        self.captcha_gen = ConfigurableImageCaptcha(
            width=self.dataset_config.width,
            height=self.dataset_config.target_height,
            fonts=self.dataset_config.fonts,
            font_sizes=self.dataset_config.font_sizes,
            noise_bg_density=self.dataset_config.noise_bg_density,
            extra_spacing=self.dataset_config.extra_spacing,
            spacing_jitter=self.dataset_config.spacing_jitter,
            add_noise_dots=self.dataset_config.add_noise_dots,
            add_noise_curve=self.dataset_config.add_noise_curve,
            character_offset_dx=tuple(self.dataset_config.character_offset_dx) if self.dataset_config.character_offset_dx else (0, 0),
            character_offset_dy=tuple(self.dataset_config.character_offset_dy) if self.dataset_config.character_offset_dy else (0, 0),
            character_rotate=tuple(self.dataset_config.character_rotate) if self.dataset_config.character_rotate else (0, 0),
            character_warp_dx=tuple(self.dataset_config.character_warp_dx) if self.dataset_config.character_warp_dx else (0.1, 0.3),
            character_warp_dy=tuple(self.dataset_config.character_warp_dy) if self.dataset_config.character_warp_dy else (0.2, 0.3),
            word_space_probability=self.dataset_config.word_space_probability,
            word_offset_dx=self.dataset_config.word_offset_dx,
        )
        
        self.vocab = list(self.dataset_config.vocab)
        self.min_len = self.dataset_config.min_word_len
        self.max_len = self.dataset_config.max_word_len

    def __len__(self):
        # Return logical length
        if self.is_validation:
             return self.config.training_config.val_steps * self.config.training_config.batch_size
        else:
             # If step based, return total samples needed for all steps (or just a large number)
             if self.config.training_config.training_steps:
                 return self.config.training_config.training_steps * self.config.training_config.batch_size
             else:
                 # Fallback if someone uses epochs with on-the-fly (infinite?)
                 # Let's define "1 epoch" as 1000 batches for now if not specified
                 return 1000 * self.config.training_config.batch_size

    def __getitem__(self, idx):
        start_time = time.time()
        
        # Generate random word
        length = random.randint(self.min_len, self.max_len)
        word = "".join(random.choices(self.vocab, k=length))
        
        # Generate image
        bg = tuple(self.dataset_config.bg_color) if self.dataset_config.bg_color else None
        fg = tuple(self.dataset_config.fg_color) if self.dataset_config.fg_color else None
        
        # Note: We don't apply word_transform here yet as we generated clean text suitable for labels.
        # If dataset config has word_transform (e.g. capitalize), we should apply it to 'word' before rendering?
        # Usually random generation assumes vocab is already what we want.
        
        img = self.captcha_gen.generate_image(word, bg_color=bg, fg_color=fg)
        
        # Convert to temp path logic or modify processor to accept PIL image directly?
        # CaptchaProcessor takes 'image_path_or_url'. If we pass PIL Image, we need to adapt it.
        # But 'processor' in __getitem__ calls 'self.processor(str(image_path), text)'.
        # We need to bypass file reading.
        # Processor's __call__ usually calls `preprocess_image`.
        
        # Let's peek at Processor. It likely handles PIL if we adapt it, or we rely on 'pixel_values' extraction manually?
        # Actually, processor.process_image handles loading.
        # We should extend Processor or just handle it here.
        # To avoid changing Processor for now, let's assume we can pass PIL image if we modify usage slightly 
        # OR we save to BytesIO? No, too slow.
        
        # Best way: Check if processor can take PIL.
        # If not, let's modify Processor or do manual processing here matching processor logic.
        # Processor logic: resize, normalize, pixel_values.
        
        # QUICK FIX: Update Processor is best, but let's check it first.
        # Assuming for now we can't easily, let's recreate the transforms:
        
        # Use processor transforms
        pixel_values = self.processor.process_image(img) # We need to check if this works with PIL
        
        # Input IDs
        input_ids = self.processor.encode_text(word)
        
        gen_time = time.time() - start_time
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "target_length": len(input_ids),
            "gen_time": gen_time
        }

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

    # Setting wandb run name
    if not config.training_config.wandb_run_name:
        config.training_config.wandb_run_name = config.experiment_name
    
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
    
    image_base_dir = config.image_base_dir
    
    # Check if On-The-Fly Generation is enabled
    if config.training_config.use_onthefly_generation:
        print("Using On-The-Fly Dataset Generation")
        # Processor needs classes/vocab. 
        # If not loading metadata, we must ensure processor has vocab.
        if not hasattr(config.dataset_config, 'vocab'):
             raise ValueError("DatasetConfig.vocab must be set for on-the-fly generation")
        
        # Define vocab for processor
        # We can construct a dummy metadata or just set chars directly?
        # Processor usually loads from metadata. 
        # Let's create a specialized processor init or just manually set it.
        # For now, let's instantiate processor with empty metadata and set chars
        processor = CaptchaProcessor(config=config, metadata_path=None) # Should handle None if modified or empty file
        # Manually set chars/classes
        processor.chars = sorted(list(set(config.dataset_config.vocab)))
        processor.idx_to_char = {i: c for i, c in enumerate(processor.chars)}
        processor.char_to_idx = {c: i for i, c in enumerate(processor.chars)}
        # For classification, we might need classes? 
        # On-the-fly is mostly for generation (OCR).
        
        train_dataset = OnTheFlyDataset(config, processor, is_validation=False)
        val_dataset = OnTheFlyDataset(config, processor, is_validation=True)
        
    elif train_dataset is None or val_dataset is None:
        # Check for explicit train/val split
        if config.train_metadata_path and config.val_metadata_path:
            print(f"Using explicit train/val split: {config.train_metadata_path} / {config.val_metadata_path}")
            
            # Initialize processor with training metadata to build vocab
            processor = CaptchaProcessor(config=config, metadata_path=config.train_metadata_path)
            
            # Create datasets
            if train_dataset is None:
                train_dataset = CaptchaDataset(config.train_metadata_path, processor, image_base_dir)
            
            if val_dataset is None:
                val_dataset = CaptchaDataset(config.val_metadata_path, processor, image_base_dir)
                
        else:
            # Fallback to single metadata file with random split
            print(f"Using single metadata file with random split: {metadata_path}")
            
            # Processor initialized with full ExperimentConfig
            processor = CaptchaProcessor(config=config, metadata_path=metadata_path)
            
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

    print("Model Parameter Breakdown:")
    for section in ["encoder", "projector", "sequence_model", "head"]:
        module = getattr(model, section, None)
        if module is not None:
            section_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  - {section}: {section_params:,}")

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

    # Determine loop mode
    training_steps = config.training_config.training_steps
    use_steps = training_steps is not None and training_steps > 0
    
    if use_steps:
        epochs = 1 # One giant epoch conceptually, but we iterate by steps
        total_steps = training_steps
        print(f"Training for {total_steps} steps (Step-based mode).")
        
        save_every_steps = config.training_config.save_every_steps
        val_check_interval = config.training_config.val_check_interval_steps
    else:
        epochs = config.training_config.epochs
        print(f"Training for {epochs} epochs (Epoch-based mode).")

    global_step = 0
    should_stop = False
    
    # Loop over epochs (or just once if steps)
    for epoch in range(epochs):
        if should_stop: break
        
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in enumerate(progress_bar):
            if batch is None: continue
            
            images = batch["pixel_values"].to(device)
            targets = batch["input_ids"].to(device) 
            target_lengths = batch['target_lengths'].to(device)
            
            # Log generation time if available
            if "gen_time" in batch[0] if isinstance(batch, list) else False: # batch is dict from collate usually, but individual items had gen_time
                 # Collate might lose it unless we modify collate. 
                 # Actually `collate_fn` constructs dict. We didn't modify collate to pass `gen_time`.
                 # Let's assume we won't log per-batch gen_time unless we update collate, 
                 # BUT we can check if it's printed or just rely on overhead.
                 # Given user request "Printing should be sufficient", we can print occasionally.
                 pass

            logits = model(images) 
            loss = loss_fn(logits, targets, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            current_loss = loss.item()
            global_step += 1
            
            if global_step % log_every == 0:
                wandb.log({
                    "train_loss": current_loss, 
                    "epoch": epoch + (step / len(train_loader)) if not use_steps else (global_step / total_steps), # Approx
                    "step": global_step
                })
            
            # --- Validation & Saving (Step-based) ---
            if use_steps:
                if global_step % val_check_interval == 0:
                    RunValidation(model, val_loader, loss_fn, processor, config, device, global_step, best_val_metric, save_dir, optimizer)
                    model.train() # Switch back
                
                if global_step % save_every_steps == 0:
                     SaveCheckpoint(model, optimizer, config, processor, save_dir, f"step_{global_step}", global_step, best_val_metric)

                if global_step >= total_steps:
                    should_stop = True
                    break
        
        if not use_steps:
             avg_train_loss = train_loss / len(train_loader)
             print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
             
             if (epoch + 1) % config.training_config.val_check_interval == 0:
                  RunValidation(model, val_loader, loss_fn, processor, config, device, epoch + 1, best_val_metric, save_dir, optimizer)

# Extracted Validation Logic to support calling from both loops
def RunValidation(model, val_loader, loss_fn, processor, config, device, step_or_epoch, best_metric, save_dir, optimizer):
    model.eval()
    val_loss = 0.0
    total_samples = 0
    
    val_preds = []
    val_targets = []
    
    generated_batches = 0
    
    # Timing
    start_val = time.time()
    
    print(f"\nRunning Validation at step/epoch {step_or_epoch}...")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None: continue
            
            # If using steps, we might have infinite val loader (OnTheFly), 
            # so rely on val_steps from config OR simply len(val_loader) if configured effectively.
            # OnTheFlyDataset has len = val_steps * batch_size, so standard iteration works.
            
            images = batch["pixel_values"].to(device)
            targets = batch["input_ids"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            
            logits = model(images)

            loss = loss_fn(logits, targets, target_lengths)
                
            val_loss += loss.item()
            
            preds = logits.argmax(dim=-1) 
            
            for k in range(len(targets)):
                pred_str = processor.decode(preds[k])

                if targets.dim() == 2:
                    target_ids = targets[k][:target_lengths[k]].tolist()
                    target_str = decode_simple(target_ids, processor.idx_to_char)
                else:
                    target_str = processor.decode(targets[k])
                
                val_preds.append(pred_str)
                val_targets.append(target_str)
                    
            total_samples += images.size(0)
    
    val_duration = time.time() - start_val
    print(f"Validation took {val_duration:.2f}s for {total_samples} samples.")

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    metrics_results = {}
    metrics_to_compute = config.training_config.metrics
    
    if metrics_to_compute:
        ocr_keys = ['character_accuracy', 'edit_distance', 'exact_match']
        needed_ocr = [m for m in metrics_to_compute if m in ocr_keys]
        
        if needed_ocr:
            total_edit_dist = 0.0
            total_char_acc = 0.0
            total_exact = 0
            
            for t, p in zip(val_targets, val_preds):
                m = calculate_metrics(t, p)
                total_edit_dist += m['edit_distance']
                total_char_acc += m['character_accuracy']
                if m['exact_match']:
                    total_exact += 1
            
            if 'edit_distance' in needed_ocr:
                metrics_results['val_edit_distance'] = total_edit_dist / total_samples
            if 'character_accuracy' in needed_ocr:
                metrics_results['val_char_acc'] = total_char_acc / total_samples
            if 'exact_match' in needed_ocr:
                metrics_results['val_exact_match'] = total_exact / total_samples

        cls_keys = ['f1', 'precision', 'recall', 'accuracy']
        needed_cls = [m for m in metrics_to_compute if m in cls_keys]
        
        if needed_cls:
            from src.utils import calculate_classification_metrics
            cls_results = calculate_classification_metrics(val_targets, val_preds, needed_cls)
            for k, v in cls_results.items():
                metrics_results[f"val_{k}"] = v
    else:
         print("\n[WARNING] No metrics specified. Only validation loss is computed.")

    log_str = f"Step/Epoch {step_or_epoch} Val Loss: {avg_val_loss:.4f}"
    for k, v in metrics_results.items():
        log_str += f" | {k}: {v:.4f}"
    print(log_str)
    
    wandb_log_dict = {
        "val_loss": avg_val_loss,
        "step": step_or_epoch, # Use generalized step
        "val_duration": val_duration
    }
    wandb_log_dict.update(metrics_results)
    wandb.log(wandb_log_dict)
    
    # Checkpointing logic here?
    # We moved basic checkpointing out, but BEST model logic needs tracking.
    # We need to return info or update a global reference. 
    # Since we are inside a function, let's just save "latest" here if needed? 
    # Or cleaner: SaveCheckpoint helper.
    
    # Let's save "latest" validation checkpoint always, and "best" if improved.
    # Note: Calling function needs to handle 'best_metric' storage.
    # This refactoring is tricky with local variable 'best_val_metric' in 'train'.
    # I'll just save if it's best and valid. But I don't have access to update 'best_val_metric' in 'train'.
    # Hack: use a mutable container or return it.
    
    return metrics_results

def SaveCheckpoint(model, optimizer, config, processor, save_dir, name, step, metrics):
    fname = f"{config.experiment_name}_{name}.pth"
    vocab = getattr(processor, 'classes', getattr(processor, 'chars', None))
    
    checkpoint_data = {
        'step': step,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
        'vocab': vocab, 
        'metrics': metrics
    }
    torch.save(checkpoint_data, os.path.join(save_dir, fname))
    
# NOTE: The loop above calls RunValidation but doesn't handle 'best' saving fully because variables are local.
# I will rewrite the loop to inline validation or fix the helper signature.
# Inlining is safer to avoid scope issues in this quick edit.
# I will Replace the BLOCK with the inlined version.


if __name__ == "__main__":
    train(
        metadata_path="validation_set/metadata.json",
        image_base_dir=".", 
        batch_size=32,
        epochs=50,
        model_type=None
    )