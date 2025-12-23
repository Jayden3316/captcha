
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_classification_metrics(targets: List[Union[str, int]], preds: List[Union[str, int]], metrics_list: List[str]) -> Dict[str, float]:
    """
    Calculate classification metrics using sklearn.
    
    Args:
        targets: List of ground truth values (strings or ints)
        preds: List of predicted values (strings or ints)
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    # Ensure inputs are lists of something comparable
    # If they are strings, sklearn can handle them for accuracy/f1/etc with average='macro' usually
    # But for precision/recall/f1, we need to handle labels carefully if not all present
    
    # Convert to string for consistency if mixed? usually they are same type
    
    if 'accuracy' in metrics_list:
        results['accuracy'] = accuracy_score(targets, preds)
        
    if 'f1' in metrics_list:
        results['f1'] = f1_score(targets, preds, average='macro', zero_division=0)
        
    if 'precision' in metrics_list:
        results['precision'] = precision_score(targets, preds, average='macro', zero_division=0)
        
    if 'recall' in metrics_list:
        results['recall'] = recall_score(targets, preds, average='macro', zero_division=0)
        
    return results
