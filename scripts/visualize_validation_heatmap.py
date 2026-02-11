#!/usr/bin/env python3
"""Generate heatmap-style validation results table.

Creates a publication-quality figure with ROI accuracy shown as a heatmap.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Color scheme for heatmap (red = poor, yellow = medium, green = good)
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


def calculate_depth_metrics(metrics: Dict, results_dir: Path = None) -> Dict[str, float]:
    """Calculate depth-stratified ROI accuracy from metrics.

    Priority:
    1. Pre-computed depth_stratified field
    2. raw_data depths + roi_correct (if depths are valid, not all zeros)
    3. Recalculate from true_positions_mm + source_coords_mm.npy
    4. per_position source_depth_mm
    """
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

    # Try raw_data (has per-trial depths and roi_correct)
    raw_data = snr_data.get('raw_data', {})
    depths = raw_data.get('depths', [])
    roi_correct = raw_data.get('roi_correct', [])
    true_positions = raw_data.get('true_positions_mm', [])

    # Check if depths are valid (not all zeros)
    depths_valid = depths and max(depths) > 0.01

    # If depths are invalid but we have true_positions, recalculate depths
    if not depths_valid and true_positions and results_dir:
        config_name = metrics.get('config_name', '')
        config_dir = results_dir / config_name
        coords_file = config_dir / 'data' / 'step3_source_coords_mm.npy'

        if coords_file.exists():
            source_coords = np.load(coords_file)
            brain_top_z = source_coords[:, 2].max()
            depths = [brain_top_z - pos[2] for pos in true_positions]
            depths_valid = True

    if depths_valid and roi_correct:
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

    # Fallback: compute from per_position source_depth_mm
    depth_correct = {k: 0 for k in depth_bins}
    depth_total = {k: 0 for k in depth_bins}

    per_position = snr_data.get('per_position', {})

    for pos_name, pos_data in per_position.items():
        depth = pos_data.get('source_depth_mm')
        if depth is None:
            continue

        for bin_name, (low, high) in depth_bins.items():
            if low <= depth < high:
                n_trials = pos_data.get('n_trials', 25)
                depth_total[bin_name] += n_trials
                depth_correct[bin_name] += n_trials * pos_data.get('roi_accuracy', 0)
                break

    depth_accuracy = {}
    for bin_name in depth_bins:
        if depth_total[bin_name] > 0:
            depth_accuracy[bin_name] = depth_correct[bin_name] / depth_total[bin_name] * 100
        else:
            depth_accuracy[bin_name] = np.nan

    return depth_accuracy


def calculate_depth_localization_error(metrics: Dict, results_dir: Path = None) -> Dict[str, float]:
    """Calculate depth-stratified localization error from metrics.

    Similar to calculate_depth_metrics but returns localization error instead of ROI accuracy.
    """
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
        depth_errors = {}
        for bin_name in depth_bins:
            bin_data = depth_strat.get(bin_name)
            if bin_data is not None:
                depth_errors[bin_name] = bin_data.get('localization_error_mean', np.nan)
            else:
                depth_errors[bin_name] = np.nan
        return depth_errors

    # Try raw_data (has per-trial depths and localization_errors)
    raw_data = snr_data.get('raw_data', {})
    depths = raw_data.get('depths', [])
    loc_errors = raw_data.get('localization_errors', [])
    true_positions = raw_data.get('true_positions_mm', [])

    # Check if depths are valid (not all zeros)
    depths_valid = depths and max(depths) > 0.01

    # If depths are invalid but we have true_positions, recalculate depths
    if not depths_valid and true_positions and results_dir:
        config_name = metrics.get('config_name', '')
        config_dir = results_dir / config_name
        coords_file = config_dir / 'data' / 'step3_source_coords_mm.npy'

        if coords_file.exists():
            source_coords = np.load(coords_file)
            brain_top_z = source_coords[:, 2].max()
            depths = [brain_top_z - pos[2] for pos in true_positions]
            depths_valid = True

    if depths_valid and loc_errors:
        depth_error_sums = {k: 0.0 for k in depth_bins}
        depth_total = {k: 0 for k in depth_bins}

        for depth, error in zip(depths, loc_errors):
            for bin_name, (low, high) in depth_bins.items():
                if low <= depth < high:
                    depth_total[bin_name] += 1
                    depth_error_sums[bin_name] += error
                    break

        depth_errors = {}
        for bin_name in depth_bins:
            if depth_total[bin_name] > 0:
                depth_errors[bin_name] = depth_error_sums[bin_name] / depth_total[bin_name]
            else:
                depth_errors[bin_name] = np.nan
        return depth_errors

    return {k: np.nan for k in depth_bins}


def create_heatmap_table(results_dir: Path, output_file: Path = None):
    """Create a heatmap-style validation results table."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    # Build data for table
    data = []
    for m in metrics_list:
        config_name = m['config_name'].replace('_coarse22', '')
        bem = m.get('bem_type', 'N/A').capitalize()
        source = m.get('source_type', 'N/A').capitalize()
        method = m.get('inverse_method', 'N/A').upper()

        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)

        depth_acc = calculate_depth_metrics(m, results_dir)

        data.append({
            'Config': config_name,
            'BEM': bem,
            'Source': source,
            'Method': method,
            'ROI %': roi_acc,
            'Error (mm)': loc_error,
            '0-1mm': depth_acc.get('0-1mm', np.nan),
            '1-2mm': depth_acc.get('1-2mm', np.nan),
            '2-3mm': depth_acc.get('2-3mm', np.nan),
            '3-4mm': depth_acc.get('3-4mm', np.nan),
            '4+mm': depth_acc.get('4+mm', np.nan),
        })

    # Sort by ROI accuracy (descending)
    data = sorted(data, key=lambda x: x['ROI %'], reverse=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis('off')

    # Define columns
    columns = ['Rank', 'Config', 'BEM', 'Source', 'Method', 'ROI %', 'Error', '0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']
    col_widths = [0.05, 0.18, 0.08, 0.10, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07]

    # Create table data
    table_data = []
    for i, row in enumerate(data):
        table_data.append([
            f"#{i+1}",
            row['Config'],
            row['BEM'],
            row['Source'],
            row['Method'],
            f"{row['ROI %']:.1f}%",
            f"{row['Error (mm)']:.2f}mm",
            f"{row['0-1mm']:.1f}%" if not np.isnan(row['0-1mm']) else 'N/A',
            f"{row['1-2mm']:.1f}%" if not np.isnan(row['1-2mm']) else 'N/A',
            f"{row['2-3mm']:.1f}%" if not np.isnan(row['2-3mm']) else 'N/A',
            f"{row['3-4mm']:.1f}%" if not np.isnan(row['3-4mm']) else 'N/A',
            f"{row['4+mm']:.1f}%" if not np.isnan(row['4+mm']) else 'N/A',
        ])

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        colWidths=col_widths,
        loc='center',
        cellLoc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color the cells based on values
    heatmap_cols = [5, 7, 8, 9, 10, 11]  # ROI %, depth columns
    error_col = 6  # Error column (inverse coloring)

    for i, row in enumerate(data):
        row_idx = i + 1  # +1 for header

        # ROI % column
        roi_val = row['ROI %']
        cell = table[(row_idx, 5)]
        color = CMAP(roi_val / 100)
        cell.set_facecolor(color)
        cell.set_text_props(color='white' if roi_val > 50 else 'black', fontweight='bold')

        # Error column (inverse: low is good)
        error_val = row['Error (mm)']
        cell = table[(row_idx, 6)]
        # Scale: 0mm = green, 6mm = red
        color = CMAP(1 - min(error_val / 6, 1))
        cell.set_facecolor(color)
        cell.set_text_props(color='white' if error_val < 3 else 'black', fontweight='bold')

        # Depth columns
        for j, key in enumerate(['0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']):
            col_idx = 7 + j
            val = row[key]
            cell = table[(row_idx, col_idx)]
            if np.isnan(val):
                cell.set_facecolor('#e0e0e0')
                cell.set_text_props(color='#666666')
            else:
                color = CMAP(val / 100)
                cell.set_facecolor(color)
                cell.set_text_props(color='white' if val > 50 else 'black', fontweight='bold')

    # Style header row
    for j, col in enumerate(columns):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Add alternating row backgrounds for non-heatmap columns
    for i in range(len(data)):
        row_idx = i + 1
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(5):  # Config, BEM, Source, Method columns
            cell = table[(row_idx, j)]
            cell.set_facecolor(bg_color)

    # Title
    ax.set_title('Validation Results: ROI Accuracy by Configuration\n(Depth-Stratified Metrics)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('ROI Accuracy (%)', fontsize=10)

    # Add annotation
    fig.text(0.5, 0.02,
             'Green = high accuracy | Red = low accuracy | Depth = distance from dorsal brain surface',
             ha='center', fontsize=10, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.05, 0.83, 0.95])

    # Save
    if output_file is None:
        output_file = results_dir / 'summary' / 'validation_heatmap.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved heatmap to {output_file}")

    # Also save as PDF for publication
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_file}")

    plt.close()

    return output_file


def create_method_comparison_chart(results_dir: Path, output_file: Path = None):
    """Create a bar chart comparing methods across BEM types."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        return

    # Organize by method and BEM
    method_bem_data = {}
    for m in metrics_list:
        method = m.get('inverse_method', 'N/A').upper()
        bem = m.get('bem_type', 'N/A')
        source = m.get('source_type', 'N/A')

        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100

        key = f"{bem}_{source}"
        if method not in method_bem_data:
            method_bem_data[method] = {}
        if key not in method_bem_data[method]:
            method_bem_data[method][key] = roi_acc

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    methods = sorted(method_bem_data.keys())
    configs = sorted(set(k for m in method_bem_data.values() for k in m.keys()))

    x = np.arange(len(configs))
    width = 0.12

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        values = [method_bem_data[method].get(c, 0) for c in configs]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i], edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_xlabel('Configuration (BEM_Source)', fontsize=12)
    ax.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax.set_title('ROI Accuracy by Inverse Method and Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=10)
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_file is None:
        output_file = results_dir / 'summary' / 'method_comparison.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved method comparison to {output_file}")
    plt.close()

    return output_file


def create_depth_analysis_chart(results_dir: Path, output_file: Path = None):
    """Create a chart showing accuracy vs depth for top configs."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        return

    # Get top 10 configs by ROI accuracy
    sorted_metrics = sorted(metrics_list,
                           key=lambda m: m.get('snr_results', {}).get('10', {}).get('roi_accuracy', {}).get('exact', 0),
                           reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(12, 8))

    depth_labels = ['0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']
    x = np.arange(len(depth_labels))
    width = 0.08

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, m in enumerate(sorted_metrics):
        config_name = m['config_name'].replace('_coarse22', '')
        depth_acc = calculate_depth_metrics(m, results_dir)
        values = [depth_acc.get(d, 0) for d in depth_labels]

        offset = (i - 5 + 0.5) * width
        ax.bar(x + offset, values, width, label=config_name, color=colors[i], edgecolor='white')

    ax.set_xlabel('Source Depth', fontsize=12)
    ax.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax.set_title('Depth-Stratified ROI Accuracy (Top 10 Configurations)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(depth_labels, fontsize=11)
    ax.legend(title='Config', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_file is None:
        output_file = results_dir / 'summary' / 'depth_analysis.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved depth analysis to {output_file}")
    plt.close()

    return output_file


def create_localization_error_heatmap(results_dir: Path, output_file: Path = None):
    """Create a heatmap showing depth-stratified localization error."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    # Build data for table
    data = []
    for m in metrics_list:
        config_name = m['config_name'].replace('_coarse22', '')
        bem = m.get('bem_type', 'N/A').capitalize()
        source = m.get('source_type', 'N/A').capitalize()
        method = m.get('inverse_method', 'N/A').upper()

        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)

        depth_error = calculate_depth_localization_error(m, results_dir)

        data.append({
            'Config': config_name,
            'BEM': bem,
            'Source': source,
            'Method': method,
            'ROI %': roi_acc,
            'Error (mm)': loc_error,
            '0-1mm': depth_error.get('0-1mm', np.nan),
            '1-2mm': depth_error.get('1-2mm', np.nan),
            '2-3mm': depth_error.get('2-3mm', np.nan),
            '3-4mm': depth_error.get('3-4mm', np.nan),
            '4+mm': depth_error.get('4+mm', np.nan),
        })

    # Sort by overall localization error (ascending - lower is better)
    data = sorted(data, key=lambda x: x['Error (mm)'])

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.axis('off')

    # Define columns
    columns = ['Rank', 'Config', 'BEM', 'Source', 'Method', 'ROI %', 'Avg Error', '0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']
    col_widths = [0.05, 0.18, 0.08, 0.10, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07]

    # Create table data
    table_data = []
    for i, row in enumerate(data):
        table_data.append([
            f"#{i+1}",
            row['Config'],
            row['BEM'],
            row['Source'],
            row['Method'],
            f"{row['ROI %']:.1f}%",
            f"{row['Error (mm)']:.2f}mm",
            f"{row['0-1mm']:.2f}" if not np.isnan(row['0-1mm']) else 'N/A',
            f"{row['1-2mm']:.2f}" if not np.isnan(row['1-2mm']) else 'N/A',
            f"{row['2-3mm']:.2f}" if not np.isnan(row['2-3mm']) else 'N/A',
            f"{row['3-4mm']:.2f}" if not np.isnan(row['3-4mm']) else 'N/A',
            f"{row['4+mm']:.2f}" if not np.isnan(row['4+mm']) else 'N/A',
        ])

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        colWidths=col_widths,
        loc='center',
        cellLoc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color the cells based on values (inverse scale: low error = green)
    # Scale: 0mm = green, 6mm = red
    ERROR_CMAP = plt.cm.RdYlGn_r  # Reversed: green for low values

    for i, row in enumerate(data):
        row_idx = i + 1  # +1 for header

        # ROI % column (normal: high = green)
        roi_val = row['ROI %']
        cell = table[(row_idx, 5)]
        color = CMAP(roi_val / 100)
        cell.set_facecolor(color)
        cell.set_text_props(color='white' if roi_val > 50 else 'black', fontweight='bold')

        # Error column (inverse: low is good)
        error_val = row['Error (mm)']
        cell = table[(row_idx, 6)]
        color = CMAP(1 - min(error_val / 6, 1))
        cell.set_facecolor(color)
        cell.set_text_props(color='white' if error_val < 3 else 'black', fontweight='bold')

        # Depth error columns (inverse: low is good)
        for j, key in enumerate(['0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']):
            col_idx = 7 + j
            val = row[key]
            cell = table[(row_idx, col_idx)]
            if np.isnan(val):
                cell.set_facecolor('#e0e0e0')
                cell.set_text_props(color='#666666')
            else:
                # Scale: 0mm = green, 6mm = red
                color = CMAP(1 - min(val / 6, 1))
                cell.set_facecolor(color)
                cell.set_text_props(color='white' if val < 3 else 'black', fontweight='bold')

    # Style header row
    for j, col in enumerate(columns):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Add alternating row backgrounds for non-heatmap columns
    for i in range(len(data)):
        row_idx = i + 1
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(5):  # Config, BEM, Source, Method columns
            cell = table[(row_idx, j)]
            cell.set_facecolor(bg_color)

    # Title
    ax.set_title('Validation Results: Localization Error by Configuration\n(Depth-Stratified Metrics, mm)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 6))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Localization Error (mm)', fontsize=10)

    # Add annotation
    fig.text(0.5, 0.02,
             'Green = low error (accurate) | Red = high error | Depth = distance from dorsal brain surface',
             ha='center', fontsize=10, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.05, 0.83, 0.95])

    # Save
    if output_file is None:
        output_file = results_dir / 'summary' / 'localization_error_heatmap.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved localization error heatmap to {output_file}")

    # Also save as PDF for publication
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_file}")

    plt.close()

    return output_file


def generate_validation_summary_md(results_dir: Path, output_file: Path = None):
    """Generate VALIDATION_SUMMARY.md with complete metrics."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        print(f"No metrics found in {results_dir}")
        return

    # Sort by ROI accuracy
    sorted_by_roi = sorted(metrics_list,
                           key=lambda m: m.get('snr_results', {}).get('10', {}).get('roi_accuracy', {}).get('exact', 0),
                           reverse=True)

    # Sort by localization error
    sorted_by_error = sorted(metrics_list,
                             key=lambda m: m.get('snr_results', {}).get('10', {}).get('localization_error', {}).get('mean', float('inf')))

    lines = []
    lines.append("# Validation Results Summary")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Configurations tested:** {len(metrics_list)}")
    lines.append(f"**Results directory:** `{results_dir}`")
    lines.append("")

    # Best performers summary
    lines.append("## Best Performers")
    lines.append("")
    lines.append("### By ROI Accuracy")
    lines.append("")
    lines.append("| Rank | Config | BEM | Source | Method | ROI % | Error (mm) |")
    lines.append("|------|--------|-----|--------|--------|-------|------------|")
    for i, m in enumerate(sorted_by_roi[:5]):
        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)
        config = m['config_name'].replace('_coarse22', '')
        bem = m.get('bem_type', 'N/A')
        source = m.get('source_type', 'N/A')
        method = m.get('inverse_method', 'N/A')
        lines.append(f"| {i+1} | {config} | {bem} | {source} | {method} | {roi_acc:.1f}% | {loc_error:.2f} |")
    lines.append("")

    lines.append("### By Localization Error")
    lines.append("")
    lines.append("| Rank | Config | BEM | Source | Method | Error (mm) | ROI % |")
    lines.append("|------|--------|-----|--------|--------|------------|-------|")
    for i, m in enumerate(sorted_by_error[:5]):
        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)
        config = m['config_name'].replace('_coarse22', '')
        bem = m.get('bem_type', 'N/A')
        source = m.get('source_type', 'N/A')
        method = m.get('inverse_method', 'N/A')
        lines.append(f"| {i+1} | {config} | {bem} | {source} | {method} | {loc_error:.2f} | {roi_acc:.1f}% |")
    lines.append("")

    # Depth-stratified ROI accuracy table
    lines.append("## Depth-Stratified ROI Accuracy")
    lines.append("")
    lines.append("| Config | Overall | 0-1mm | 1-2mm | 2-3mm | 3-4mm | 4+mm |")
    lines.append("|--------|---------|-------|-------|-------|-------|------|")
    for m in sorted_by_roi:
        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        depth_acc = calculate_depth_metrics(m, results_dir)
        config = m['config_name'].replace('_coarse22', '')

        depth_vals = []
        for d in ['0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']:
            v = depth_acc.get(d, np.nan)
            depth_vals.append(f"{v:.1f}%" if not np.isnan(v) else "N/A")

        lines.append(f"| {config} | {roi_acc:.1f}% | {' | '.join(depth_vals)} |")
    lines.append("")

    # Depth-stratified localization error table
    lines.append("## Depth-Stratified Localization Error (mm)")
    lines.append("")
    lines.append("| Config | Overall | 0-1mm | 1-2mm | 2-3mm | 3-4mm | 4+mm |")
    lines.append("|--------|---------|-------|-------|-------|-------|------|")
    for m in sorted_by_error:
        snr_data = m.get('snr_results', {}).get('10', {})
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)
        depth_error = calculate_depth_localization_error(m, results_dir)
        config = m['config_name'].replace('_coarse22', '')

        depth_vals = []
        for d in ['0-1mm', '1-2mm', '2-3mm', '3-4mm', '4+mm']:
            v = depth_error.get(d, np.nan)
            depth_vals.append(f"{v:.2f}" if not np.isnan(v) else "N/A")

        lines.append(f"| {config} | {loc_error:.2f} | {' | '.join(depth_vals)} |")
    lines.append("")

    # Method comparison
    lines.append("## Method Comparison")
    lines.append("")
    method_stats = {}
    for m in metrics_list:
        method = m.get('inverse_method', 'N/A')
        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)
        if method not in method_stats:
            method_stats[method] = {'roi': [], 'error': []}
        method_stats[method]['roi'].append(roi_acc)
        method_stats[method]['error'].append(loc_error)

    lines.append("| Method | Avg ROI % | Std | Avg Error | Std |")
    lines.append("|--------|-----------|-----|-----------|-----|")
    for method, stats in sorted(method_stats.items(), key=lambda x: np.mean(x[1]['roi']), reverse=True):
        roi_mean = np.mean(stats['roi'])
        roi_std = np.std(stats['roi'])
        error_mean = np.mean(stats['error'])
        error_std = np.std(stats['error'])
        lines.append(f"| {method} | {roi_mean:.1f}% | ±{roi_std:.1f} | {error_mean:.2f}mm | ±{error_std:.2f} |")
    lines.append("")

    # BEM + Source type comparison
    lines.append("## BEM + Source Type Comparison")
    lines.append("")
    bem_source_stats = {}
    for m in metrics_list:
        bem = m.get('bem_type', 'N/A')
        source = m.get('source_type', 'N/A')
        key = f"{bem}_{source}"
        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)
        if key not in bem_source_stats:
            bem_source_stats[key] = {'roi': [], 'error': []}
        bem_source_stats[key]['roi'].append(roi_acc)
        bem_source_stats[key]['error'].append(loc_error)

    lines.append("| BEM + Source | Avg ROI % | Avg Error |")
    lines.append("|--------------|-----------|-----------|")
    for key, stats in sorted(bem_source_stats.items(), key=lambda x: np.mean(x[1]['roi']), reverse=True):
        roi_mean = np.mean(stats['roi'])
        error_mean = np.mean(stats['error'])
        lines.append(f"| {key} | {roi_mean:.1f}% | {error_mean:.2f}mm |")
    lines.append("")

    # Write to file
    if output_file is None:
        output_file = results_dir.parent / 'VALIDATION_SUMMARY.md'

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved VALIDATION_SUMMARY.md to {output_file}")
    return output_file


def create_summary_dashboard(results_dir: Path, output_file: Path = None):
    """Create a multi-panel summary dashboard."""
    metrics_list = load_metrics(results_dir)

    if not metrics_list:
        return

    fig = plt.figure(figsize=(20, 16))

    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)

    # Panel 1: Top 10 performers (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_metrics = sorted(metrics_list,
                           key=lambda m: m.get('snr_results', {}).get('10', {}).get('roi_accuracy', {}).get('exact', 0),
                           reverse=True)[:10]

    configs = [m['config_name'].replace('_coarse22', '').replace('V', '').replace('_', '\n', 1) for m in sorted_metrics]
    roi_accs = [m.get('snr_results', {}).get('10', {}).get('roi_accuracy', {}).get('exact', 0) * 100 for m in sorted_metrics]

    colors = [CMAP(v/100) for v in roi_accs]
    bars = ax1.barh(range(len(configs)), roi_accs, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels(configs, fontsize=9)
    ax1.set_xlabel('ROI Accuracy (%)')
    ax1.set_title('Top 10 Configurations by ROI Accuracy', fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, roi_accs)):
        ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Panel 2: Localization error vs ROI accuracy scatter
    ax2 = fig.add_subplot(gs[0, 1])

    for m in metrics_list:
        snr_data = m.get('snr_results', {}).get('10', {})
        roi_acc = snr_data.get('roi_accuracy', {}).get('exact', 0) * 100
        loc_error = snr_data.get('localization_error', {}).get('mean', 0)

        bem = m.get('bem_type', '')
        source = m.get('source_type', '')

        marker = 'o' if bem == 'sphere' else 's'
        color = {'volumetric': 'blue', 'surface': 'green', 'roi_based': 'red'}.get(source, 'gray')

        ax2.scatter(loc_error, roi_acc, marker=marker, c=color, s=100, alpha=0.7, edgecolor='white')

    ax2.set_xlabel('Localization Error (mm)')
    ax2.set_ylabel('ROI Accuracy (%)')
    ax2.set_title('Localization Error vs ROI Accuracy', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Sphere'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Ellipsoid'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Volumetric'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Surface'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='ROI-based'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Panel 3: Method comparison
    ax3 = fig.add_subplot(gs[1, 0])

    method_accs = {}
    for m in metrics_list:
        method = m.get('inverse_method', 'N/A')
        roi_acc = m.get('snr_results', {}).get('10', {}).get('roi_accuracy', {}).get('exact', 0) * 100
        if method not in method_accs:
            method_accs[method] = []
        method_accs[method].append(roi_acc)

    methods = list(method_accs.keys())
    means = [np.mean(method_accs[m]) for m in methods]
    stds = [np.std(method_accs[m]) for m in methods]

    colors_method = [CMAP(m/100) for m in means]
    bars = ax3.bar(methods, means, yerr=stds, color=colors_method, edgecolor='white', capsize=5)
    ax3.set_ylabel('ROI Accuracy (%)')
    ax3.set_title('Average ROI Accuracy by Inverse Method', fontweight='bold')
    ax3.set_ylim(0, 100)

    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.1f}%', ha='center', fontweight='bold')

    # Panel 4: BEM type comparison
    ax4 = fig.add_subplot(gs[1, 1])

    bem_source_accs = {}
    for m in metrics_list:
        bem = m.get('bem_type', 'N/A')
        source = m.get('source_type', 'N/A')
        key = f"{bem}\n{source}"
        roi_acc = m.get('snr_results', {}).get('10', {}).get('roi_accuracy', {}).get('exact', 0) * 100
        if key not in bem_source_accs:
            bem_source_accs[key] = []
        bem_source_accs[key].append(roi_acc)

    keys = list(bem_source_accs.keys())
    means = [np.mean(bem_source_accs[k]) for k in keys]
    stds = [np.std(bem_source_accs[k]) for k in keys]

    colors_bem = [CMAP(m/100) for m in means]
    bars = ax4.bar(keys, means, yerr=stds, color=colors_bem, edgecolor='white', capsize=5)
    ax4.set_ylabel('ROI Accuracy (%)')
    ax4.set_title('Average ROI Accuracy by BEM + Source Type', fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.tick_params(axis='x', labelsize=9)

    for bar, val in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

    # Panel 5 & 6: Depth analysis for best config
    ax5 = fig.add_subplot(gs[2, :])

    # Get top 6 configs
    top_configs = sorted_metrics[:6]

    depth_labels = ['0-1mm\n(superficial)', '1-2mm', '2-3mm', '3-4mm', '4+mm\n(deep)']
    x = np.arange(len(depth_labels))
    width = 0.12

    colors_top = plt.cm.Set1(np.linspace(0, 1, 6))

    for i, m in enumerate(top_configs):
        config_name = m['config_name'].replace('_coarse22', '')
        depth_acc = calculate_depth_metrics(m, results_dir)
        values = [depth_acc.get(d.split('\n')[0], 0) for d in depth_labels]

        offset = (i - 3 + 0.5) * width
        ax5.bar(x + offset, values, width, label=config_name, color=colors_top[i], edgecolor='white')

    ax5.set_xlabel('Source Depth (from dorsal brain surface)', fontsize=12)
    ax5.set_ylabel('ROI Accuracy (%)', fontsize=12)
    ax5.set_title('Depth-Stratified ROI Accuracy for Top 6 Configurations', fontweight='bold', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(depth_labels, fontsize=11)
    ax5.legend(title='Config', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax5.set_ylim(0, 100)
    ax5.grid(axis='y', alpha=0.3)

    # Overall title
    fig.suptitle('Mouse EEG Source Localization: Validation Campaign Results\n(26 Configurations, 127 Test Positions, 25 Trials/Position)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_file is None:
        output_file = results_dir / 'summary' / 'validation_dashboard.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved dashboard to {output_file}")

    # PDF version
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_file}")

    plt.close()

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate validation heatmap visualizations')
    parser.add_argument('--results-dir', type=Path,
                       default=Path(__file__).parent.parent / 'validation' / 'results' / 'original',
                       help='Directory containing validation results')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for figures')

    args = parser.parse_args()

    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = args.results_dir / 'summary'

    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    print(f"Saving figures to: {output_base}")
    print()

    # Generate all visualizations
    create_heatmap_table(args.results_dir, output_base / 'validation_heatmap.png')
    create_localization_error_heatmap(args.results_dir, output_base / 'localization_error_heatmap.png')
    create_method_comparison_chart(args.results_dir, output_base / 'method_comparison.png')
    create_depth_analysis_chart(args.results_dir, output_base / 'depth_analysis.png')
    create_summary_dashboard(args.results_dir, output_base / 'validation_dashboard.png')
    generate_validation_summary_md(args.results_dir)

    print("\nAll visualizations generated!")
