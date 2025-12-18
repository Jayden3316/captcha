#!/usr/bin/env python3
"""
Tesseract OCR Baseline Testing Script
Tests Tesseract OCR on a validation dataset and computes accuracy metrics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import time

import pytesseract
from PIL import Image

from utils import extract_text_tesseract, calculate_edit_distance, calculate_metrics, upsample_image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata.json containing image information."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def run_tesseract_evaluation(
    validation_dir: str,
    metadata_filename: str = 'metadata.json',
    tesseract_config: str = '--psm 7',
    verbose: bool = True,
    target_width: int = None,
    target_height: int = None,
    upsample: bool = False
) -> Tuple[Dict, List[Dict]]:
    """
    Run Tesseract OCR on the validation dataset and compute metrics.
    
    Args:
        validation_dir: Path to validation_set directory
        metadata_filename: Name of metadata file (default: 'metadata.json')
        tesseract_config: Tesseract configuration string
        verbose: Print progress information
    
    Returns:
        Tuple of (aggregate_metrics, detailed_results)
    """
    validation_path = Path(validation_dir)
    metadata_path = validation_path / metadata_filename
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    metadata = load_metadata(str(metadata_path))
    
    if verbose:
        print(f"Loaded {len(metadata)} images from metadata")
        print(f"Using Tesseract config: {tesseract_config}")
        if upsample:
            upsample_info = []
            if target_width:
                upsample_info.append(f"width={target_width}")
            if target_height:
                upsample_info.append(f"height={target_height}")
            print(f"Image upsampling: {'x'.join(upsample_info)} (maintaining aspect ratio)")
        else:
            print("Image upsampling: disabled")
        print("-" * 80)
    
    # Track metrics
    results = []
    total_exact_matches = 0
    total_case_insensitive_matches = 0
    total_edit_distance = 0
    total_character_accuracy = 0.0
    processing_times = []
    
    # Process each image
    for idx, record in enumerate(metadata):
        image_path = record['image_path']
        ground_truth = record['word_rendered']
        
        # Handle relative paths
        if not Path(image_path).is_absolute():
            image_path = validation_path / image_path
        
        if not Path(image_path).exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Run OCR
        start_time = time.time()
        prediction = extract_text_tesseract(
            str(image_path), 
            config=tesseract_config, 
            target_height=target_height, 
            target_width=target_width, 
            upsample=upsample
        )
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Calculate metrics
        metrics = calculate_metrics(ground_truth, prediction)
        
        # Update aggregate statistics
        total_exact_matches += int(metrics['exact_match'])
        total_case_insensitive_matches += int(metrics['case_insensitive_match'])
        total_edit_distance += metrics['edit_distance']
        total_character_accuracy += metrics['character_accuracy']
        
        # Store detailed result
        result = {
            'image_path': str(image_path),
            'ground_truth': ground_truth,
            'prediction': prediction,
            'word_length': record['word_length'],
            'processing_time': processing_time,
            **metrics
        }
        results.append(result)
        
        # Print progress
        if verbose and (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(metadata)} images...")
    
    # Calculate aggregate metrics
    num_samples = len(results)
    aggregate_metrics = {
        'total_samples': num_samples,
        'exact_match_accuracy': total_exact_matches / num_samples if num_samples > 0 else 0,
        'case_insensitive_accuracy': total_case_insensitive_matches / num_samples if num_samples > 0 else 0,
        'average_edit_distance': total_edit_distance / num_samples if num_samples > 0 else 0,
        'average_character_accuracy': total_character_accuracy / num_samples if num_samples > 0 else 0,
        'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
        'tesseract_config': tesseract_config,
        'upsample': upsample,
        'target_width': target_width,
        'target_height': target_height
    }
    
    return aggregate_metrics, results


def print_summary(aggregate_metrics: Dict, detailed_results: List[Dict]):
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 80)
    print("TESSERACT OCR BASELINE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Samples:                {aggregate_metrics['total_samples']}")
    print(f"Exact Match Accuracy:         {aggregate_metrics['exact_match_accuracy']:.2%}")
    print(f"Case-Insensitive Accuracy:    {aggregate_metrics['case_insensitive_accuracy']:.2%}")
    print(f"Average Character Accuracy:   {aggregate_metrics['average_character_accuracy']:.2%}")
    print(f"Average Edit Distance:        {aggregate_metrics['average_edit_distance']:.3f}")
    print(f"Average Processing Time:      {aggregate_metrics['average_processing_time']*1000:.2f} ms")
    print(f"Tesseract Config:             {aggregate_metrics['tesseract_config']}")
    print("=" * 80)
    
    if aggregate_metrics.get('upsample', False):
        upsample_info = []
        if aggregate_metrics.get('target_width'):
            upsample_info.append(f"width={aggregate_metrics['target_width']}")
        if aggregate_metrics.get('target_height'):
            upsample_info.append(f"height={aggregate_metrics['target_height']}")
        print(f"Image Upsampling:              {'x'.join(upsample_info)} (aspect ratio maintained)")
    else:
        print(f"Image Upsampling:              disabled")

    print('='*80)

    # Show some examples of errors
    errors = [r for r in detailed_results if not r['exact_match']]
    if errors:
        print(f"\nShowing first 10 errors (out of {len(errors)} total):")
        print("-" * 80)
        for i, error in enumerate(errors[:10]):
            print(f"\nError {i+1}:")
            print(f"  Ground Truth: '{error['ground_truth']}'")
            print(f"  Prediction:   '{error['prediction']}'")
            print(f"  Edit Distance: {error['edit_distance']}")
            print(f"  Image: {Path(error['image_path']).name}")


def save_results(aggregate_metrics: Dict, detailed_results: List[Dict], output_dir: str = "."):
    """Save results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save aggregate metrics
    metrics_file = output_path / "tesseract_baseline_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"\nSaved aggregate metrics to: {metrics_file}")
    
    # Save detailed results
    results_file = output_path / "tesseract_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Saved detailed results to: {results_file}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test Tesseract OCR baseline on validation dataset'
    )
    parser.add_argument(
        '--validation-dir',
        type=str,
        default='validation_set',
        help='Path to validation_set directory (default: validation_set)'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default='metadata.json',
        help='Name of metadata file (default: metadata.json)'
    )
    parser.add_argument(
        '--psm',
        type=int,
        default=8,
        help='Tesseract Page Segmentation Mode'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save results (default: current directory)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '--upsample',
        action='store_true',
        help='Enable image upsampling before OCR'
    )
    parser.add_argument(
        '--target-width',
        type=int,
        default=None,
        help='Target width for upsampling (maintains aspect ratio). Requires --upsample'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=None,
        help='Target height for upsampling (maintains aspect ratio). Requires --upsample'
    )
    
    args = parser.parse_args()

    if args.upsample and args.target_width is None and args.target_height is None:
        parser.error("--upsample requires at least one of --target-width or --target-height")
    
    # Build Tesseract config
    tesseract_config = f'--psm {args.psm}'
    
    print("Starting Tesseract OCR baseline evaluation...")
    print(f"Validation directory: {args.validation_dir}")
    
    try:
        # Run evaluation
        aggregate_metrics, detailed_results = run_tesseract_evaluation(
            validation_dir=args.validation_dir,
            metadata_filename=args.metadata_file,
            tesseract_config=tesseract_config,
            verbose=not args.quiet,
            target_width=args.target_width,
            target_height=args.target_height,
            upsample=args.upsample
        )
        
        # Print summary
        print_summary(aggregate_metrics, detailed_results)
        
        # Save results
        save_results(aggregate_metrics, detailed_results, args.output_dir)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
