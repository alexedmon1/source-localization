#!/usr/bin/env python
"""
Parameter sweep for shell-based source space optimization.

Tests different configurations to find optimal balance between:
- Source count (coverage)
- Conditioning (inverse stability)
- Depth distribution (resolution at different depths)

Configurations tested:
- Number of shells: 4 vs 6
- Density: sparse (~150), medium (~250), dense (~400)
- Shell scale ranges: default vs more superficial

Usage:
    python scripts/011_shell_parameter_sweep.py

Output:
    Parameter sweep results table and recommendations
"""

import sys
from pathlib import Path

# Add source to path if running directly
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import numpy as np
from source_localization.config import Config
from source_localization.steps import electrode_registration, bem_model, source_space, forward_solution
from source_localization.source_space import shell_based


def compute_conditioning(fwd_obj):
    """Compute forward matrix conditioning metrics."""
    G = fwd_obj['sol']['data']
    n_channels, n_cols = G.shape

    U, s, Vh = np.linalg.svd(G, full_matrices=False)

    condition_number = s[0] / s[-1] if s[-1] > 0 else np.inf
    threshold = 0.01 * s[0]
    effective_rank = np.sum(s > threshold)

    return {
        'condition_number': condition_number,
        'effective_rank': effective_rank,
        'n_channels': n_channels,
        'singular_values': s,
    }


def compute_depth_distribution(source_coords_mm, bem_params):
    """Analyze source distribution by depth."""
    center = np.array(bem_params['center_mm'])
    semi_axes = np.array(bem_params['semi_axes_mm'])

    # Compute normalized ellipsoid distance (0=center, 1=surface)
    centered = source_coords_mm - center
    ellipsoid_dists = np.sqrt(np.sum((centered / semi_axes) ** 2, axis=1))

    # Bin into depth quartiles
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    hist, _ = np.histogram(ellipsoid_dists, bins=bins)

    return {
        'mean_depth': np.mean(ellipsoid_dists),
        'std_depth': np.std(ellipsoid_dists),
        'min_depth': np.min(ellipsoid_dists),
        'max_depth': np.max(ellipsoid_dists),
        'depth_quartiles': hist.tolist(),
        'depth_values': ellipsoid_dists,
    }


def run_config(config_name, shell_config, base_config, bem_outputs, elec_outputs):
    """Run a single shell configuration and return metrics."""
    print(f"\n  Testing: {config_name}")

    # Clone config and update shell settings
    config = Config.from_preset('shell_based_ellipsoid')
    config['source_space']['shell_based'].update(shell_config)
    config['outputs']['save_intermediate'] = False
    config['outputs']['dir'] = '/tmp/shell_sweep'

    # Merge previous outputs
    previous = {**elec_outputs, **bem_outputs}

    # Create source space
    src_outputs = source_space.run(config, previous)
    previous.update(src_outputs)

    # Compute forward solution
    fwd_outputs = forward_solution.run(config, previous)

    # Compute metrics
    cond_metrics = compute_conditioning(fwd_outputs['fwd'])
    depth_metrics = compute_depth_distribution(
        src_outputs['source_coords_mm'],
        bem_outputs['bem_params']
    )

    return {
        'config_name': config_name,
        'n_sources': src_outputs['n_sources'],
        'n_shells': shell_config.get('n_shells', 4),
        'condition_number': cond_metrics['condition_number'],
        'effective_rank': cond_metrics['effective_rank'],
        'mean_depth': depth_metrics['mean_depth'],
        'depth_quartiles': depth_metrics['depth_quartiles'],
        'shell_config': shell_config,
    }


def main():
    print("="*70)
    print("SHELL-BASED SOURCE SPACE PARAMETER SWEEP")
    print("="*70)

    # First, set up electrode registration and BEM (shared across all configs)
    print("\nSetting up shared pipeline components...")

    base_config = Config.from_preset('shell_based_ellipsoid')
    base_config['outputs']['save_intermediate'] = False
    base_config['outputs']['dir'] = '/tmp/shell_sweep'

    # Electrode registration
    elec_outputs = electrode_registration.run(base_config, {})

    # BEM model
    bem_outputs = bem_model.run(base_config, elec_outputs)

    print("\n" + "="*70)
    print("RUNNING PARAMETER SWEEP")
    print("="*70)

    # Define configurations to test
    configs = [
        # Sparse configurations (~150 sources)
        ('4-shell-sparse', {
            'n_shells': 4,
            'shell_scales': [0.3, 0.5, 0.7, 0.9],
            'min_points_per_shell': 15,
            'max_points_per_shell': 60,
            'scale_by_area': True,
        }),
        ('6-shell-sparse', {
            'n_shells': 6,
            'shell_scales': [0.2, 0.35, 0.5, 0.65, 0.8, 0.95],
            'min_points_per_shell': 10,
            'max_points_per_shell': 40,
            'scale_by_area': True,
        }),

        # Medium configurations (~250 sources) - current default
        ('4-shell-medium', {
            'n_shells': 4,
            'shell_scales': [0.3, 0.5, 0.7, 0.9],
            'min_points_per_shell': 20,
            'max_points_per_shell': 100,
            'scale_by_area': True,
        }),
        ('6-shell-medium', {
            'n_shells': 6,
            'shell_scales': [0.2, 0.35, 0.5, 0.65, 0.8, 0.95],
            'min_points_per_shell': 15,
            'max_points_per_shell': 60,
            'scale_by_area': True,
        }),

        # Dense configurations (~400 sources)
        ('4-shell-dense', {
            'n_shells': 4,
            'shell_scales': [0.3, 0.5, 0.7, 0.9],
            'min_points_per_shell': 40,
            'max_points_per_shell': 160,
            'scale_by_area': True,
        }),
        ('6-shell-dense', {
            'n_shells': 6,
            'shell_scales': [0.2, 0.35, 0.5, 0.65, 0.8, 0.95],
            'min_points_per_shell': 25,
            'max_points_per_shell': 100,
            'scale_by_area': True,
        }),

        # Superficial-biased (more sources near surface)
        ('4-shell-superficial', {
            'n_shells': 4,
            'shell_scales': [0.5, 0.7, 0.85, 0.95],
            'min_points_per_shell': 20,
            'max_points_per_shell': 100,
            'scale_by_area': True,
        }),

        # Uniform density (no area scaling)
        ('4-shell-uniform', {
            'n_shells': 4,
            'shell_scales': [0.3, 0.5, 0.7, 0.9],
            'min_points_per_shell': 50,
            'max_points_per_shell': 50,
            'scale_by_area': False,
        }),
    ]

    results = []
    for config_name, shell_config in configs:
        try:
            result = run_config(
                config_name, shell_config,
                base_config, bem_outputs, elec_outputs
            )
            results.append(result)
            print(f"    Sources: {result['n_sources']}, "
                  f"Condition #: {result['condition_number']:.1f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary table
    print("\n" + "="*80)
    print("PARAMETER SWEEP RESULTS")
    print("="*80)
    print(f"{'Configuration':<22} {'Sources':>8} {'Shells':>7} {'Cond #':>10} "
          f"{'Eff Rank':>10} {'Mean Depth':>11}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x['condition_number']):
        print(f"{r['config_name']:<22} {r['n_sources']:>8} {r['n_shells']:>7} "
              f"{r['condition_number']:>10.1f} {r['effective_rank']:>5}/30    "
              f"{r['mean_depth']:>10.2f}")

    print("-"*80)

    # Analysis
    print("\nANALYSIS:")

    # Best overall
    best = min(results, key=lambda x: x['condition_number'])
    print(f"\n1. Best conditioning: {best['config_name']}")
    print(f"   - Condition #: {best['condition_number']:.1f}")
    print(f"   - Sources: {best['n_sources']}")

    # Best for ~200 sources (comparable to ROI-based)
    medium_results = [r for r in results if 150 <= r['n_sources'] <= 300]
    if medium_results:
        best_medium = min(medium_results, key=lambda x: x['condition_number'])
        print(f"\n2. Best for ~200 sources: {best_medium['config_name']}")
        print(f"   - Condition #: {best_medium['condition_number']:.1f}")
        print(f"   - Sources: {best_medium['n_sources']}")

    # Conditioning vs source count tradeoff
    print("\n3. Conditioning vs Source Count:")
    sparse = [r for r in results if r['n_sources'] < 180]
    medium = [r for r in results if 180 <= r['n_sources'] < 350]
    dense = [r for r in results if r['n_sources'] >= 350]

    if sparse:
        avg_sparse = np.mean([r['condition_number'] for r in sparse])
        print(f"   - Sparse (<180 sources): avg condition # = {avg_sparse:.1f}")
    if medium:
        avg_medium = np.mean([r['condition_number'] for r in medium])
        print(f"   - Medium (180-350 sources): avg condition # = {avg_medium:.1f}")
    if dense:
        avg_dense = np.mean([r['condition_number'] for r in dense])
        print(f"   - Dense (>350 sources): avg condition # = {avg_dense:.1f}")

    # 4 vs 6 shells
    print("\n4. Number of Shells Comparison:")
    four_shell = [r for r in results if r['n_shells'] == 4]
    six_shell = [r for r in results if r['n_shells'] == 6]

    if four_shell:
        avg_4 = np.mean([r['condition_number'] for r in four_shell])
        print(f"   - 4 shells: avg condition # = {avg_4:.1f}")
    if six_shell:
        avg_6 = np.mean([r['condition_number'] for r in six_shell])
        print(f"   - 6 shells: avg condition # = {avg_6:.1f}")

    # Depth distribution
    print("\n5. Depth Distribution (mean normalized depth):")
    for r in sorted(results, key=lambda x: x['mean_depth'])[:3]:
        print(f"   - {r['config_name']}: {r['mean_depth']:.2f} "
              f"(quartiles: {r['depth_quartiles']})")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("""
Based on the sweep results:

1. For BEST CONDITIONING (inverse stability):
   - Use sparse configurations with 4 shells
   - Tradeoff: fewer sources = less spatial resolution

2. For BALANCED PERFORMANCE (recommended):
   - 4-shell-medium or 6-shell-medium
   - ~200-250 sources with good conditioning

3. For MAXIMUM RESOLUTION:
   - Use dense configurations
   - Accept higher condition number (still better than volumetric)

4. For SUPERFICIAL FOCUS (cortical activity):
   - Use superficial-biased configuration
   - More sources near surface where EEG resolution is best
""")


if __name__ == '__main__':
    main()
