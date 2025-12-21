import torch
import torch.nn as nn
from typing import Dict, Any

class CTCLossWrapper(nn.Module):
    def __init__(self, blank=0, zero_infinity=True):
        super().__init__()
        self.loss_fn = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Adapts standard [Batch, Seq, Vocab] logits for CTCLoss which expects [Seq, Batch, Vocab].
        """
        # 1. Permute to [Seq_Len, Batch, Vocab] for PyTorch CTCLoss
        logits_time_major = logits.permute(1, 0, 2)
        log_probs = logits_time_major.log_softmax(dim=2)
        
        T, N, _ = log_probs.shape
        # Input lengths are full sequence length for each item in batch
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=logits.device)
        
        return self.loss_fn(log_probs, targets, input_lengths, target_lengths)

class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Flattens logits and targets for standard CrossEntropyLoss.
        Note: target_lengths is unused here but kept for interface consistency.
        """
        # logits: [Batch, Seq, Vocab] -> [Batch * Seq, Vocab]
        # targets: [Batch, Seq] -> [Batch * Seq]
        return self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

def get_loss_function(config) -> nn.Module:
    """
    Factory for loss functions. 
    Config expects 'loss_type' attribute (present in ModelConfig).
    """
    if hasattr(config, 'loss_type'):
        if config.loss_type == 'ctc':
            return CTCLossWrapper(blank=0)
        elif config.loss_type == 'cross_entropy':
            return CrossEntropyLossWrapper(ignore_index=0)
            
    # Fallback to defaults if strictly needed, but ModelConfig always has loss_type
    return CTCLossWrapper(blank=0)