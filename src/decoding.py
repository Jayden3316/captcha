from typing import List, Union
import torch

def decode_simple(token_ids: List[int], vocab: List[str]) -> str:
    """
    Simple autoregressive/classification decoding.
    Directly maps token IDs to characters.
    """
    return "".join([vocab[t] for t in token_ids if t < len(vocab)])

def decode_ctc(token_ids: List[int], vocab: Union[List[str], dict], blank_idx: int = 0) -> str:
    """
    Decode sequence using Connectionist Temporal Classification (CTC) rules.
    1. Collapse repeated characters.
    2. Remove blanks.
    """
    decoded = []
    last_token = -1
    
    for token in token_ids:
        if token == blank_idx:
            last_token = blank_idx
            continue
            
        if token != last_token:
            decoded.append(token)
            
        last_token = token
        
    if isinstance(vocab, dict):
        return "".join([vocab.get(t, "") for t in decoded])
    else:
        # Legacy list behavior (assuming 1-based indexing for chars)
        return "".join([vocab[t - 1] for t in decoded if t - 1 < len(vocab) and t > 0])

def efficient_decode_batch_ctc(batch_preds: torch.Tensor, vocab: List[str], blank_idx: int = 0) -> List[str]:
    """
    Efficiently decode a batch of CTC predictions.
    batch_preds: [batch_size, seq_len] of token IDs
    """
    results = []
    batch_preds_list = batch_preds.tolist()
    for seq in batch_preds_list:
        results.append(decode_ctc(seq, vocab, blank_idx))
    return results
