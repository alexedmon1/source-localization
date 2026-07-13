"""
Validation summary module for aggregating results from existing folders.

This module provides functions to read validation results from result directories
and generate comprehensive comparison reports including depth-based metrics.

Usage:
    # CLI
    source-localization validate --summarize /path/to/validation/results

    # Python API
    from source_localization.validation.summarize import summarize_validation_results
    summary = summarize_validation_results('/path/to/validation/results')
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def load_metrics_from_folder(result_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load metrics.json from a validation result folder.

    Parameters
    ----------
    result_dir : Path
        Path to a single validation result directory (e.g., P01_ellipsoid_surface_sloreta/)

    Returns
    -------
    metrics : dict or None
        Loaded metrics data, or None if not found
    """
    metrics_file = result_dir / 'metrics.json'
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)


def summarize_validation_folders(
    results_dir: str,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Summarize validation results from multiple result folders.

    Parameters
    ----------
    results_dir : str
        Path to validation results directory (containing P01_*, P02_*, etc.)
    output_file : str, optional
        Path to save summary report (markdown). If None, uses results_dir/SUMMARY.md
    verbose : bool
        Print progress and summary

    Returns
    -------
    summary : dict
        Aggregated summary with depth-based metrics
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_path}")

    # Find all result folders
    result_folders = sorted([
        d for d in results_path.iterdir()
        if d.is_dir() and (d / 'metrics.json').exists()
    ])

    if not result_folders:
        raise ValueError(f"No result folders with metrics.json found in: {results_path}")

    if verbose:
        print(f"Found {len(result_folders)} validation results")

    # Load all metrics
    all_results = {}
    for folder in result_folders:
        metrics = load_metrics_from_folder(folder)
        if metrics:
            all_results[folder.name] = metrics

    # Build summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'results_dir': str(results_path),
        'n_configs': len(all_results),
        'configs': [],
        'by_roi_accuracy': [],
        'by_localization_error': [],
        'by_method': {},
        'by_bem_type': {},
        'by_source_type': {},
        'depth_analysis': {}
    }

    # Extract key metrics from each config
    for config_name, metrics in all_results.items():
        # Get SNR results (typically just SNR=10)
        snr_results = metrics.get('snr_results', {})
        snr_key = list(snr_results.keys())[0] if snr_results else None

        if not snr_key:
            continue

        snr_data = snr_results[snr_key]

        config_summary = {
            'name': config_name,
            'pipeline_name': metrics.get('pipeline_name', config_name),
            'bem_type': metrics.get('bem_type', 'unknown'),
            'source_type': metrics.get('source_type', 'unknown'),
            'inverse_method': metrics.get('inverse_method', 'unknown'),
            'n_sources': metrics.get('n_sources', 0),
            'n_simulations': snr_data.get('n_simulations', 0),
            'snr_db': snr_data.get('snr_db', 10),
            'localization_error': {
                'mean': snr_data.get('localization_error', {}).get('mean', 0),
                'std': snr_data.get('localization_error', {}).get('std', 0),
                'median': snr_data.get('localization_error', {}).get('median', 0),
                'min': snr_data.get('localization_error', {}).get('min', 0),
                'max': snr_data.get('localization_error', {}).get('max', 0),
            },
            'roi_accuracy': snr_data.get('roi_accuracy', {}).get('exact', 0),
            'depth_bias': snr_data.get('depth_bias', {}),
            'depth_stratified': snr_data.get('depth_stratified', {})
        }

        summary['configs'].append(config_summary)

        # Group by method
        method = config_summary['inverse_method']
        if method not in summary['by_method']:
            summary['by_method'][method] = []
        summary['by_method'][method].append(config_summary)

        # Group by BEM type
        bem = config_summary['bem_type']
        if bem not in summary['by_bem_type']:
            summary['by_bem_type'][bem] = []
        summary['by_bem_type'][bem].append(config_summary)

        # Group by source type
        src = config_summary['source_type']
        if src not in summary['by_source_type']:
            summary['by_source_type'][src] = []
        summary['by_source_type'][src].append(config_summary)

    # Sort by metrics
    summary['by_roi_accuracy'] = sorted(
        summary['configs'],
        key=lambda x: x['roi_accuracy'],
        reverse=True
    )
    summary['by_localization_error'] = sorted(
        summary['configs'],
        key=lambda x: x['localization_error']['mean']
    )

    # Aggregate depth analysis
    depth_bins = ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']
    for bin_name in depth_bins:
        summary['depth_analysis'][bin_name] = {
            'best_config': None,
            'best_error': float('inf'),
            'best_roi_accuracy': 0,
            'all_configs': []
        }

        for cfg in summary['configs']:
            depth_data = cfg.get('depth_stratified', {}).get(bin_name, {})
            if depth_data:
                error = depth_data.get('localization_error_mean', float('inf'))
                roi_acc = depth_data.get('roi_accuracy', 0)

                summary['depth_analysis'][bin_name]['all_configs'].append({
                    'name': cfg['name'],
                    'error': error,
                    'roi_accuracy': roi_acc,
                    'n_trials': depth_data.get('n_trials', 0)
                })

                if error < summary['depth_analysis'][bin_name]['best_error']:
                    summary['depth_analysis'][bin_name]['best_error'] = error
                    summary['depth_analysis'][bin_name]['best_config'] = cfg['name']
                    summary['depth_analysis'][bin_name]['best_roi_accuracy'] = roi_acc

    # Method comparison
    summary['method_comparison'] = {}
    for method, configs in summary['by_method'].items():
        errors = [c['localization_error']['mean'] for c in configs]
        accuracies = [c['roi_accuracy'] for c in configs]
        summary['method_comparison'][method] = {
            'n_configs': len(configs),
            'avg_error': np.mean(errors),
            'avg_roi_accuracy': np.mean(accuracies),
            'best_error': min(errors),
            'best_accuracy': max(accuracies)
        }

    # Generate report
    if output_file is None:
        output_file = results_path / 'VALIDATION_SUMMARY.md'

    _generate_summary_report(summary, output_file)

    if verbose:
        _print_summary(summary)

    return summary


def _generate_summary_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Generate markdown summary report."""

    with open(output_path, 'w') as f:
        f.write("# Validation Summary Report\n\n")
        f.write(f"**Generated:** {summary['timestamp']}\n")
        f.write(f"**Results Directory:** {summary['results_dir']}\n")
        f.write(f"**Configurations:** {summary['n_configs']}\n\n")

        # Top 10 by ROI Accuracy
        f.write("## Top 10 by ROI Accuracy\n\n")
        f.write("| Rank | Config | BEM | Source | Method | Sources | Error (mm) | ROI Acc |\n")
        f.write("|------|--------|-----|--------|--------|---------|------------|----------|\n")
        for i, cfg in enumerate(summary['by_roi_accuracy'][:10], 1):
            f.write(f"| {i} | {cfg['name']} | {cfg['bem_type']} | {cfg['source_type']} | "
                   f"{cfg['inverse_method']} | {cfg['n_sources']} | "
                   f"{cfg['localization_error']['mean']:.2f} | {cfg['roi_accuracy']:.1%} |\n")
        f.write("\n")

        # Top 10 by Localization Error
        f.write("## Top 10 by Localization Error\n\n")
        f.write("| Rank | Config | Error (mm) | Median | ROI Acc | Method |\n")
        f.write("|------|--------|------------|--------|---------|--------|\n")
        for i, cfg in enumerate(summary['by_localization_error'][:10], 1):
            f.write(f"| {i} | {cfg['name']} | {cfg['localization_error']['mean']:.2f} | "
                   f"{cfg['localization_error']['median']:.2f} | {cfg['roi_accuracy']:.1%} | "
                   f"{cfg['inverse_method']} |\n")
        f.write("\n")

        # Method Comparison
        f.write("## Inverse Method Comparison\n\n")
        f.write("| Method | N Configs | Avg Error (mm) | Avg ROI Acc | Best Error | Best Acc |\n")
        f.write("|--------|-----------|----------------|-------------|------------|----------|\n")
        for method, stats in sorted(summary['method_comparison'].items(),
                                    key=lambda x: x[1]['avg_roi_accuracy'], reverse=True):
            f.write(f"| {method} | {stats['n_configs']} | {stats['avg_error']:.2f} | "
                   f"{stats['avg_roi_accuracy']:.1%} | {stats['best_error']:.2f} | "
                   f"{stats['best_accuracy']:.1%} |\n")
        f.write("\n")

        # Depth-Based Analysis
        f.write("## Depth-Based Analysis\n\n")
        f.write("Best configuration at each depth level:\n\n")
        f.write("| Depth | Best Config | Error (mm) | ROI Acc | N Trials |\n")
        f.write("|-------|-------------|------------|---------|----------|\n")
        for depth_bin in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
            depth_data = summary['depth_analysis'].get(depth_bin, {})
            if depth_data.get('best_config'):
                f.write(f"| {depth_bin} | {depth_data['best_config']} | "
                       f"{depth_data['best_error']:.2f} | {depth_data['best_roi_accuracy']:.1%} | "
                       f"- |\n")
        f.write("\n")

        # Depth Error by Method (sLORETA only for clarity)
        f.write("### Depth Error Comparison (sLORETA configs)\n\n")
        sloreta_configs = [c for c in summary['configs'] if c['inverse_method'] == 'sLORETA']
        if sloreta_configs:
            f.write("| Config | 1-2mm | 2-3mm | 3-4mm | 4-5mm | 5+mm |\n")
            f.write("|--------|-------|-------|-------|-------|------|\n")
            for cfg in sorted(sloreta_configs, key=lambda x: x['roi_accuracy'], reverse=True)[:10]:
                row = f"| {cfg['name'].replace('_sloreta', '')} |"
                for depth_bin in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    depth_data = cfg.get('depth_stratified', {}).get(depth_bin, {})
                    if depth_data:
                        row += f" {depth_data.get('localization_error_mean', 0):.2f} |"
                    else:
                        row += " - |"
                f.write(row + "\n")
        f.write("\n")

        # Depth ROI Accuracy by Method (sLORETA only)
        f.write("### Depth ROI Accuracy Comparison (sLORETA configs)\n\n")
        if sloreta_configs:
            f.write("| Config | 1-2mm | 2-3mm | 3-4mm | 4-5mm | 5+mm |\n")
            f.write("|--------|-------|-------|-------|-------|------|\n")
            for cfg in sorted(sloreta_configs, key=lambda x: x['roi_accuracy'], reverse=True)[:10]:
                row = f"| {cfg['name'].replace('_sloreta', '')} |"
                for depth_bin in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
                    depth_data = cfg.get('depth_stratified', {}).get(depth_bin, {})
                    if depth_data:
                        row += f" {depth_data.get('roi_accuracy', 0):.1%} |"
                    else:
                        row += " - |"
                f.write(row + "\n")
        f.write("\n")

        # Depth Bias Analysis
        f.write("## Depth Bias Analysis\n\n")
        f.write("Correlation between source depth and localization error:\n\n")
        f.write("| Config | Correlation (r) | Slope (mm/mm) | Superficial Err | Deep Err |\n")
        f.write("|--------|-----------------|---------------|-----------------|----------|\n")
        for cfg in sorted(summary['configs'], key=lambda x: abs(x.get('depth_bias', {}).get('correlation', 0)))[:15]:
            bias = cfg.get('depth_bias', {})
            if bias:
                f.write(f"| {cfg['name']} | {bias.get('correlation', 0):.3f} | "
                       f"{bias.get('slope_mm_per_mm', 0):.3f} | "
                       f"{bias.get('superficial_error_mm', 0):.2f} | "
                       f"{bias.get('deep_error_mm', 0):.2f} |\n")
        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        best_roi = summary['by_roi_accuracy'][0] if summary['by_roi_accuracy'] else None
        best_error = summary['by_localization_error'][0] if summary['by_localization_error'] else None

        if best_roi:
            f.write(f"**Best for ROI-level statistics:** `{best_roi['name']}`\n")
            f.write(f"- ROI Accuracy: {best_roi['roi_accuracy']:.1%}\n")
            f.write(f"- Localization Error: {best_roi['localization_error']['mean']:.2f} mm\n\n")

        if best_error:
            f.write(f"**Best for spatial localization:** `{best_error['name']}`\n")
            f.write(f"- Localization Error: {best_error['localization_error']['mean']:.2f} mm\n")
            f.write(f"- ROI Accuracy: {best_error['roi_accuracy']:.1%}\n\n")

        # Best sLORETA config
        sloreta_best = max(sloreta_configs, key=lambda x: x['roi_accuracy']) if sloreta_configs else None
        if sloreta_best:
            f.write(f"**Recommended (sLORETA):** `{sloreta_best['name']}`\n")

    print(f"\nSummary report saved to: {output_path}")


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Configurations: {summary['n_configs']}")
    print()

    print("Top 5 by ROI Accuracy:")
    print("-" * 50)
    for i, cfg in enumerate(summary['by_roi_accuracy'][:5], 1):
        print(f"  {i}. {cfg['name']}: {cfg['roi_accuracy']:.1%} "
              f"(error: {cfg['localization_error']['mean']:.2f}mm)")
    print()

    print("Top 5 by Localization Error:")
    print("-" * 50)
    for i, cfg in enumerate(summary['by_localization_error'][:5], 1):
        print(f"  {i}. {cfg['name']}: {cfg['localization_error']['mean']:.2f}mm "
              f"(ROI: {cfg['roi_accuracy']:.1%})")
    print()

    print("Method Comparison:")
    print("-" * 50)
    for method, stats in sorted(summary['method_comparison'].items(),
                                key=lambda x: x[1]['avg_roi_accuracy'], reverse=True):
        print(f"  {method}: {stats['avg_roi_accuracy']:.1%} accuracy, "
              f"{stats['avg_error']:.2f}mm error (n={stats['n_configs']})")
    print()

    print("Best at Each Depth:")
    print("-" * 50)
    for depth_bin in ['1-2mm', '2-3mm', '3-4mm', '4-5mm', '5+mm']:
        depth_data = summary['depth_analysis'].get(depth_bin, {})
        if depth_data.get('best_config'):
            print(f"  {depth_bin}: {depth_data['best_config']} "
                  f"({depth_data['best_error']:.2f}mm, {depth_data['best_roi_accuracy']:.1%})")
