#!/usr/bin/env python3
"""Generate visualizations for dipole_size validation tests.

Compares low-noise (D01-D05) and high-noise (D06-D10) results to show
SNR effects on localization accuracy.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import re

# Color scheme
CMAP = plt.cm.RdYlGn

def load_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all metrics.json files from validation results."""
    metrics_list = []

    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name == 'summary':
            continue

        metrics_file = config_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                metrics_list.append(metrics)

    return metrics_list


def parse_config_name(config_name: str) -> Dict[str, Any]:
    """Parse config name to extract test parameters."""
    # Remove _coarse22 suffix
    name = config_name.replace('_coarse22', '')

    # Try pattern with noise type: D{id}_V{pipeline}_{amplitude}nAm_{noise}
    match = re.match(r'D(\d+)_V(\d+)_(\d+)nAm_(\w+)', name)
    if match:
        d_id = int(match.group(1))
        pipeline = f"V{match.group(2)}"
        amplitude = int(match.group(3))
        noise_type = match.group(4)

        return {
            'd_id': d_id,
            'pipeline': pipeline,
            'amplitude': amplitude,
            'noise_type': noise_type,
            'is_highnoise': noise_type == 'highnoise'
        }

    # Try pattern without noise type (old configs D01-D05 are low noise)
    # Pattern: D{id}_V{pipeline}_{amplitude}nAm
    match = re.match(r'D(\d+)_V(\d+)_(\d+)nAm$', name)
    if match:
        d_id = int(match.group(1))
        pipeline = f"V{match.group(2)}"
        amplitude = int(match.group(3))

        return {
            'd_id': d_id,
            'pipeline': pipeline,
            'amplitude': amplitude,
            'noise_type': 'lownoise',  # Old configs are low noise
            'is_highnoise': False
        }

    return None


def calculate_depth_metrics(metrics: Dict, results_dir: Path = None) -> Dict[str, float]:
    """Calculate depth-stratified ROI accuracy from metrics."""
    depth_bins = {
        '0-1mm': (0, 1),
        '1-2mm': (1, 2),
        '2-3mm': (2, 3),
        '3-4mm': (3, 4),
        '4+mm': (4, float('inf'))
    }

    snr_data = metrics.get('snr_results', {}).get('10', {})

    # Try pre-computed depth_stratified first
    depth_strat = snr_data.get('depth_stratified', {})
    if depth_strat:
        depth_accuracy = {}
        for bin_name in depth_bins:
            bin_data = depth_strat.get(bin_name)
            if bin_data is not None:
                depth_accuracy[bin_name] = bin_data.get('roi_accuracy', 0) * 100
            else:
                depth_accuracy[bin_name] = np.nan
        return depth_accuracy

    # Fallback to raw_data
    raw_data = snr_data.get('raw_data', {})
    depths = raw_data.get('depths', [])
    roi_correct = raw_data.get('roi_correct', [])

    if depths and roi_correct and max(depths) > 0.01:
        depth_correct = {k: 0 for k in depth_bins}
        depth_total = {k: 0 for k in depth_bins}

        for depth, correct in zip(depths, roi_correct):
            for bin_name, (low, high) in depth_bins.items():
                if low <= depth < high:
                    depth_total[bin_name] += 1
                    if correct:
                        depth_correct[bin_name] += 1
                    break

        depth_accuracy = {}
        for bin_name in depth_bins:
            if depth_total[bin_name] > 0:
                depth_accuracy[bin_name] = (depth_correct[bin_name] / depth_total[bin_name]) * 100
            else:
                depth_accuracy[bin_name] = np.nan
        return depth_accuracy

    return {k: np.nan for k in depth_bins}


def create_snr_effect_plot(results_dir: Path, output_file: Path = None):
    """Create plot showing ROI accuracy vs dipole amplitude for each noise level."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    # Organize data by pipeline and noise type
    data = {}
    for m in metrics_list:
        parsed = parse_config_name(m['config_name'])
        if not parsed:
            continue

        pipeline = parsed['pipeline']
        amplitude = parsed['amplitude']
        noise_type = parsed['noise_type']

        key = (pipeline, noise_type)
        if key not in data:
            data[key] = {'amplitudes': [], 'roi_acc': [], 'loc_error': []}

        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)

        data[key]['amplitudes'].append(amplitude)
        data[key]['roi_acc'].append(roi_acc)
        data[key]['loc_error'].append(loc_error)

    # Sort each series by amplitude
    for key in data:
        sorted_idx = np.argsort(data[key]['amplitudes'])
        data[key]['amplitudes'] = [data[key]['amplitudes'][i] for i in sorted_idx]
        data[key]['roi_acc'] = [data[key]['roi_acc'][i] for i in sorted_idx]
        data[key]['loc_error'] = [data[key]['loc_error'][i] for i in sorted_idx]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Define pipeline colors and styles
    pipeline_colors = {'V24': '#1f77b4', 'V03': '#ff7f0e', 'V15': '#2ca02c', 'V21': '#d62728'}
    pipeline_names = {
        'V24': 'V24 (ellipsoid+roi+sLORETA)',
        'V03': 'V03 (sphere+vol+sLORETA)',
        'V15': 'V15 (ellipsoid+roi+MNE)',
        'V21': 'V21 (sphere+roi+sLORETA)'
    }

    # Panel 1: Low-noise ROI accuracy vs amplitude
    ax1 = axes[0, 0]
    for pipeline in ['V24', 'V03', 'V15', 'V21']:
        key = (pipeline, 'lownoise')
        if key in data:
            ax1.plot(data[key]['amplitudes'], data[key]['roi_acc'],
                    'o-', color=pipeline_colors[pipeline],
                    label=pipeline_names[pipeline], linewidth=2, markersize=8)

    ax1.set_xlabel('Dipole Amplitude (nAm)', fontsize=12)
    ax1.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax1.set_title('Low Noise (1.0 µV²)\nSNR range: ~10-36 dB', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xscale('log')
    ax1.set_xticks([10, 25, 50, 100, 200])
    ax1.set_xticklabels(['10', '25', '50', '100', '200'])

    # Panel 2: High-noise ROI accuracy vs amplitude
    ax2 = axes[0, 1]
    for pipeline in ['V24', 'V03', 'V15', 'V21']:
        key = (pipeline, 'highnoise')
        if key in data:
            ax2.plot(data[key]['amplitudes'], data[key]['roi_acc'],
                    'o-', color=pipeline_colors[pipeline],
                    label=pipeline_names[pipeline], linewidth=2, markersize=8)

    ax2.set_xlabel('Dipole Amplitude (nAm)', fontsize=12)
    ax2.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax2.set_title('High Noise (25.0 µV²)\nSNR range: ~-4 to +22 dB', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xscale('log')
    ax2.set_xticks([10, 25, 50, 100, 200])
    ax2.set_xticklabels(['10', '25', '50', '100', '200'])

    # Panel 3: Low-noise Localization error vs amplitude
    ax3 = axes[1, 0]
    for pipeline in ['V24', 'V03', 'V15', 'V21']:
        key = (pipeline, 'lownoise')
        if key in data:
            ax3.plot(data[key]['amplitudes'], data[key]['loc_error'],
                    's--', color=pipeline_colors[pipeline],
                    label=pipeline_names[pipeline], linewidth=2, markersize=8)

    ax3.set_xlabel('Dipole Amplitude (nAm)', fontsize=12)
    ax3.set_ylabel('Localization Error (mm)', fontsize=12)
    ax3.set_title('Low Noise (1.0 µV²)', fontweight='bold', fontsize=12)
    ax3.set_ylim(0, 6)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    ax3.set_xscale('log')
    ax3.set_xticks([10, 25, 50, 100, 200])
    ax3.set_xticklabels(['10', '25', '50', '100', '200'])

    # Panel 4: High-noise Localization error vs amplitude
    ax4 = axes[1, 1]
    for pipeline in ['V24', 'V03', 'V15', 'V21']:
        key = (pipeline, 'highnoise')
        if key in data:
            ax4.plot(data[key]['amplitudes'], data[key]['loc_error'],
                    's--', color=pipeline_colors[pipeline],
                    label=pipeline_names[pipeline], linewidth=2, markersize=8)

    ax4.set_xlabel('Dipole Amplitude (nAm)', fontsize=12)
    ax4.set_ylabel('Localization Error (mm)', fontsize=12)
    ax4.set_title('High Noise (25.0 µV²)', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, 6)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.set_xscale('log')
    ax4.set_xticks([10, 25, 50, 100, 200])
    ax4.set_xticklabels(['10', '25', '50', '100', '200'])

    fig.suptitle('Dipole Size Validation: Effect of Signal Amplitude on Localization Accuracy\n(25 trials per position, 127 positions per config)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file is None:
        output_file = results_dir / 'summary' / 'snr_effect_plot.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved SNR effect plot to {output_file}")

    # PDF version
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_file}")

    plt.close()
    return output_file


def create_noise_comparison_table(results_dir: Path, output_file: Path = None):
    """Create table comparing low-noise vs high-noise results."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    # Organize by pipeline and amplitude
    data = {}
    for m in metrics_list:
        parsed = parse_config_name(m['config_name'])
        if not parsed:
            continue

        pipeline = parsed['pipeline']
        amplitude = parsed['amplitude']
        noise_type = parsed['noise_type']

        key = (pipeline, amplitude)
        if key not in data:
            data[key] = {}

        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)

        data[key][noise_type] = {'roi_acc': roi_acc, 'loc_error': loc_error}

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis('off')

    # Build table
    columns = ['Pipeline', 'Amplitude',
               'ROI % (low)', 'ROI % (high)', 'Δ ROI',
               'Error (low)', 'Error (high)', 'Δ Error']

    table_data = []

    for (pipeline, amplitude) in sorted(data.keys()):
        low = data[(pipeline, amplitude)].get('lownoise', {})
        high = data[(pipeline, amplitude)].get('highnoise', {})

        low_roi = low.get('roi_acc', np.nan)
        high_roi = high.get('roi_acc', np.nan)
        delta_roi = high_roi - low_roi if not (np.isnan(low_roi) or np.isnan(high_roi)) else np.nan

        low_err = low.get('loc_error', np.nan)
        high_err = high.get('loc_error', np.nan)
        delta_err = high_err - low_err if not (np.isnan(low_err) or np.isnan(high_err)) else np.nan

        table_data.append([
            pipeline,
            f"{amplitude} nAm",
            f"{low_roi:.1f}%" if not np.isnan(low_roi) else "N/A",
            f"{high_roi:.1f}%" if not np.isnan(high_roi) else "N/A",
            f"{delta_roi:+.1f}%" if not np.isnan(delta_roi) else "N/A",
            f"{low_err:.2f}mm" if not np.isnan(low_err) else "N/A",
            f"{high_err:.2f}mm" if not np.isnan(high_err) else "N/A",
            f"{delta_err:+.2f}mm" if not np.isnan(delta_err) else "N/A",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header
    for j, col in enumerate(columns):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Color delta columns
    for i, row in enumerate(table_data):
        row_idx = i + 1

        # Delta ROI column (negative = worse)
        delta_roi_str = row[4]
        if delta_roi_str != "N/A":
            delta_val = float(delta_roi_str.replace('%', '').replace('+', ''))
            cell = table[(row_idx, 4)]
            if delta_val < -5:
                cell.set_facecolor('#ffcccc')  # Red for significant drop
            elif delta_val > 5:
                cell.set_facecolor('#ccffcc')  # Green for improvement
            cell.set_text_props(fontweight='bold')

        # Delta Error column (positive = worse)
        delta_err_str = row[7]
        if delta_err_str != "N/A":
            delta_val = float(delta_err_str.replace('mm', '').replace('+', ''))
            cell = table[(row_idx, 7)]
            if delta_val > 0.3:
                cell.set_facecolor('#ffcccc')  # Red for significant increase
            elif delta_val < -0.3:
                cell.set_facecolor('#ccffcc')  # Green for improvement
            cell.set_text_props(fontweight='bold')

    ax.set_title('Dipole Size Test: Low-Noise vs High-Noise Comparison\n(Low: 1.0 µV², High: 25.0 µV²)',
                 fontsize=16, fontweight='bold', pad=20)

    fig.text(0.5, 0.02,
             'Δ ROI: negative = accuracy dropped with higher noise | Δ Error: positive = error increased with higher noise',
             ha='center', fontsize=10, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_file is None:
        output_file = results_dir / 'summary' / 'noise_comparison_table.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved noise comparison table to {output_file}")

    plt.close()
    return output_file


def create_depth_comparison_plot(results_dir: Path, output_file: Path = None):
    """Create depth-stratified comparison between low and high noise for best pipeline."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    # Get V24 results (best pipeline)
    v24_data = {}
    for m in metrics_list:
        parsed = parse_config_name(m['config_name'])
        if not parsed or parsed['pipeline'] != 'V24':
            continue

        amplitude = parsed['amplitude']
        noise_type = parsed['noise_type']

        depth_acc = calculate_depth_metrics(m, results_dir)
        v24_data[(amplitude, noise_type)] = depth_acc

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    depth_labels = ['0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']
    x = np.arange(len(depth_labels))
    width = 0.15

    amplitudes = [10, 25, 50, 100, 200]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(amplitudes)))

    # Low noise panel
    ax1 = axes[0]
    for i, amp in enumerate(amplitudes):
        key = (amp, 'lownoise')
        if key in v24_data:
            values = [v24_data[key].get(d, 0) for d in depth_labels]
            offset = (i - 2) * width
            ax1.bar(x + offset, values, width, label=f'{amp} nAm', color=colors[i])

    ax1.set_xlabel('Source Depth', fontsize=12)
    ax1.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax1.set_title('V24 (Best Pipeline) - Low Noise (1.0 µV²)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(depth_labels)
    ax1.set_ylim(0, 100)
    ax1.legend(title='Amplitude')
    ax1.grid(axis='y', alpha=0.3)

    # High noise panel
    ax2 = axes[1]
    for i, amp in enumerate(amplitudes):
        key = (amp, 'highnoise')
        if key in v24_data:
            values = [v24_data[key].get(d, 0) for d in depth_labels]
            offset = (i - 2) * width
            ax2.bar(x + offset, values, width, label=f'{amp} nAm', color=colors[i])

    ax2.set_xlabel('Source Depth', fontsize=12)
    ax2.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax2.set_title('V24 (Best Pipeline) - High Noise (25.0 µV²)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(depth_labels)
    ax2.set_ylim(0, 100)
    ax2.legend(title='Amplitude')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Depth-Stratified ROI Accuracy: Effect of Noise Level on V24 Pipeline',
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file is None:
        output_file = results_dir / 'summary' / 'depth_comparison.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved depth comparison to {output_file}")

    plt.close()
    return output_file


def generate_dipole_size_summary_md(results_dir: Path, output_file: Path = None):
    """Generate markdown summary for dipole_size tests."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    lines = []
    lines.append("# Dipole Size Validation Results")
    lines.append("")
    lines.append(f"**Generated:** {np.datetime64('now')}")
    lines.append(f"**Configurations tested:** {len(metrics_list)}")
    lines.append("")

    lines.append("## Test Design")
    lines.append("")
    lines.append("| Noise Level | Variance | Amplitudes | Expected SNR Range |")
    lines.append("|-------------|----------|------------|-------------------|")
    lines.append("| Low | 1.0 µV² | 10-200 nAm | ~10 to ~36 dB |")
    lines.append("| High | 25.0 µV² | 10-200 nAm | ~-4 to ~22 dB |")
    lines.append("")

    lines.append("## Pipelines Tested")
    lines.append("")
    lines.append("| Pipeline | BEM | Source | Inverse |")
    lines.append("|----------|-----|--------|---------|")
    lines.append("| V24 | Ellipsoid | ROI-based | sLORETA |")
    lines.append("| V21 | Sphere | ROI-based | sLORETA |")
    lines.append("| V15 | Ellipsoid | ROI-based | MNE |")
    lines.append("| V03 | Sphere | Volumetric | sLORETA |")
    lines.append("")

    # Organize data
    data = {}
    for m in metrics_list:
        parsed = parse_config_name(m['config_name'])
        if not parsed:
            continue

        pipeline = parsed['pipeline']
        amplitude = parsed['amplitude']
        noise_type = parsed['noise_type']

        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)

        data[(pipeline, amplitude, noise_type)] = {
            'roi_acc': roi_acc,
            'loc_error': loc_error
        }

    lines.append("## Results: Low Noise (1.0 µV²)")
    lines.append("")
    lines.append("| Pipeline | 10 nAm | 25 nAm | 50 nAm | 100 nAm | 200 nAm |")
    lines.append("|----------|--------|--------|--------|---------|---------|")

    for pipeline in ['V24', 'V21', 'V15', 'V03']:
        row = [pipeline]
        for amp in [10, 25, 50, 100, 200]:
            d = data.get((pipeline, amp, 'lownoise'), {})
            roi = d.get('roi_acc', np.nan)
            row.append(f"{roi:.1f}%" if not np.isnan(roi) else "N/A")
        lines.append(f"| {' | '.join(row)} |")
    lines.append("")

    lines.append("## Results: High Noise (25.0 µV²)")
    lines.append("")
    lines.append("| Pipeline | 10 nAm | 25 nAm | 50 nAm | 100 nAm | 200 nAm |")
    lines.append("|----------|--------|--------|--------|---------|---------|")

    for pipeline in ['V24', 'V21', 'V15', 'V03']:
        row = [pipeline]
        for amp in [10, 25, 50, 100, 200]:
            d = data.get((pipeline, amp, 'highnoise'), {})
            roi = d.get('roi_acc', np.nan)
            row.append(f"{roi:.1f}%" if not np.isnan(roi) else "N/A")
        lines.append(f"| {' | '.join(row)} |")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")

    # Calculate degradation for V24
    v24_low_10 = data.get(('V24', 10, 'lownoise'), {}).get('roi_acc', np.nan)
    v24_high_10 = data.get(('V24', 10, 'highnoise'), {}).get('roi_acc', np.nan)
    v24_low_200 = data.get(('V24', 200, 'lownoise'), {}).get('roi_acc', np.nan)
    v24_high_200 = data.get(('V24', 200, 'highnoise'), {}).get('roi_acc', np.nan)

    lines.append(f"1. **V24 at lowest SNR (10 nAm, high noise):** {v24_high_10:.1f}% ROI accuracy")
    lines.append(f"2. **V24 at highest SNR (200 nAm, low noise):** {v24_low_200:.1f}% ROI accuracy")

    if not np.isnan(v24_low_10) and not np.isnan(v24_high_10):
        delta = v24_low_10 - v24_high_10
        lines.append(f"3. **SNR effect at 10 nAm:** {delta:.1f}% drop from low to high noise")

    lines.append("")

    if output_file is None:
        output_file = results_dir / 'summary' / 'DIPOLE_SIZE_SUMMARY.md'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved summary to {output_file}")
    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate dipole_size validation visualizations')
    parser.add_argument('--results-dir', type=Path,
                       default=Path(__file__).parent.parent / 'validation' / 'results' / 'dipole_size',
                       help='Directory containing dipole_size validation results')

    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")

    # Generate all visualizations
    create_snr_effect_plot(args.results_dir)
    create_noise_comparison_table(args.results_dir)
    create_depth_comparison_plot(args.results_dir)
    generate_dipole_size_summary_md(args.results_dir)

    print("\nAll dipole_size visualizations generated!")
