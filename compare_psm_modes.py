#!/usr/bin/env python3
"""
Compare different Tesseract PSM (Page Segmentation Mode) configurations
to find the best baseline for your dataset.
"""

import json
from pathlib import Path
from test_tesseract_baseline import run_tesseract_evaluation, print_summary

# PSM modes to test
PSM_MODES = [
    (7, "Treat the image as a single text line"),
    (8, "Treat the image as a single word"),
    (13, "Raw line (single text line, bypass hacks)"),
    (6, "Assume a single uniform block of text"),
    (3, "Fully automatic page segmentation"),
]

def compare_psm_modes(validation_dir: str = 'validation_set'):
    """Run Tesseract with different PSM modes and compare results."""
    
    print("=" * 80)
    print("COMPARING TESSERACT PSM MODES")
    print("=" * 80)
    print(f"Dataset: {validation_dir}\n")
    
    results_comparison = []
    
    for psm_mode, description in PSM_MODES:
        print(f"\n{'=' * 80}")
        print(f"Testing PSM {psm_mode}: {description}")
        print('=' * 80)
        
        config = f'--psm {psm_mode}'
        
        try:
            aggregate_metrics, detailed_results = run_tesseract_evaluation(
                validation_dir=validation_dir,
                tesseract_config=config,
                verbose=False
            )
            
            # Store comparison data
            results_comparison.append({
                'psm_mode': psm_mode,
                'description': description,
                'metrics': aggregate_metrics
            })
            
            # Print key metrics
            print(f"\nResults for PSM {psm_mode}:")
            print(f"  Exact Match Accuracy:      {aggregate_metrics['exact_match_accuracy']:.2%}")
            print(f"  Case-Insensitive Accuracy: {aggregate_metrics['case_insensitive_accuracy']:.2%}")
            print(f"  Character Accuracy:        {aggregate_metrics['average_character_accuracy']:.2%}")
            print(f"  Avg Edit Distance:         {aggregate_metrics['average_edit_distance']:.3f}")
            print(f"  Avg Processing Time:       {aggregate_metrics['average_processing_time']*1000:.2f} ms")
            
        except Exception as e:
            print(f"Error with PSM {psm_mode}: {e}")
            continue
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'PSM':<5} {'Description':<45} {'Exact':<10} {'Case-Ins':<10} {'Char Acc':<10}")
    print("-" * 80)
    
    for result in results_comparison:
        psm = result['psm_mode']
        desc = result['description'][:43]
        metrics = result['metrics']
        exact = f"{metrics['exact_match_accuracy']:.2%}"
        case_ins = f"{metrics['case_insensitive_accuracy']:.2%}"
        char_acc = f"{metrics['average_character_accuracy']:.2%}"
        
        print(f"{psm:<5} {desc:<45} {exact:<10} {case_ins:<10} {char_acc:<10}")
    
    # Find best configuration
    best_exact = max(results_comparison, key=lambda x: x['metrics']['exact_match_accuracy'])
    best_char = max(results_comparison, key=lambda x: x['metrics']['average_character_accuracy'])
    fastest = min(results_comparison, key=lambda x: x['metrics']['average_processing_time'])
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"Best Exact Match Accuracy:  PSM {best_exact['psm_mode']} "
          f"({best_exact['metrics']['exact_match_accuracy']:.2%})")
    print(f"Best Character Accuracy:    PSM {best_char['psm_mode']} "
          f"({best_char['metrics']['average_character_accuracy']:.2%})")
    print(f"Fastest Processing:         PSM {fastest['psm_mode']} "
          f"({fastest['metrics']['average_processing_time']*1000:.2f} ms)")
    
    # Save comparison results
    output_file = Path('tesseract_psm_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(results_comparison, f, indent=2)
    print(f"\nDetailed comparison saved to: {output_file}")
    
    return results_comparison


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare different Tesseract PSM modes'
    )
    parser.add_argument(
        '--validation-dir',
        type=str,
        default='validation_set',
        help='Path to validation_set directory (default: validation_set)'
    )
    
    args = parser.parse_args()
    
    try:
        compare_psm_modes(args.validation_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
