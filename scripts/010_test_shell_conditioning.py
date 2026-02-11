#!/usr/bin/env python
"""
Test forward matrix conditioning for shell-based source space.

Compares condition number and effective rank across source space types:
- volumetric (Cartesian grid)
- surface (icosphere)
- roi_based (ROI centroids)
- shell_based (concentric shells) <- NEW

This is Phase 6 from PLAN_shell_based_source_space.md:
"Run basic validation (conditioning, source positions)"

Usage:
    python scripts/010_test_shell_conditioning.py

Output:
    Conditioning comparison table for all source space types
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


def compute_forward_conditioning(fwd, src_outputs):
    """
    Compute conditioning metrics for forward matrix.

    Parameters
    ----------
    fwd : dict
        Forward solution from forward_solution step
    src_outputs : dict
        Source space outputs containing n_sources

    Returns
    -------
    dict
        Conditioning metrics
    """
    # Get the forward matrix (leadfield)
    fwd_obj = fwd['fwd']
    G = fwd_obj['sol']['data']  # Shape: (n_channels, n_sources * n_orient)

    n_channels, n_cols = G.shape
    n_sources = src_outputs['n_sources']

    # Compute SVD
    U, s, Vh = np.linalg.svd(G, full_matrices=False)

    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 0 else np.inf

    # Effective rank (number of singular values > 1% of max)
    threshold = 0.01 * s[0]
    effective_rank = np.sum(s > threshold)

    # Also compute at 0.1% threshold
    threshold_strict = 0.001 * s[0]
    effective_rank_strict = np.sum(s > threshold_strict)

    return {
        'n_sources': n_sources,
        'n_channels': n_channels,
        'condition_number': condition_number,
        'effective_rank_1pct': effective_rank,
        'effective_rank_0.1pct': effective_rank_strict,
        'rank_ratio': effective_rank / n_channels,
        'singular_values': s,
    }


def run_pipeline_to_forward(preset_name):
    """
    Run pipeline through forward solution step.

    Returns source space and forward solution outputs.
    """
    print(f"\n{'='*60}")
    print(f"Testing preset: {preset_name}")
    print('='*60)

    # Load config
    config = Config.from_preset(preset_name)

    # Override to not save intermediate files
    config['outputs']['save_intermediate'] = False
    config['outputs']['dir'] = '/tmp/conditioning_test'

    # Step 1: Electrode registration (creates MNE info)
    elec_outputs = electrode_registration.run(config, {})

    # Step 2: BEM model
    bem_outputs = bem_model.run(config, elec_outputs)

    # Merge outputs
    previous = {**elec_outputs, **bem_outputs}

    # Step 3: Source space
    src_outputs = source_space.run(config, previous)
    previous.update(src_outputs)

    # Step 4: Forward solution
    fwd_outputs = forward_solution.run(config, previous)

    return src_outputs, fwd_outputs


def main():
    # Presets to compare
    presets = [
        # Volumetric (baseline - should have worst conditioning)
        ('ellipsoid_volumetric', 'Volumetric (Cartesian)'),
        # Surface
        ('ellipsoid_surface', 'Surface (icosphere)'),
        # ROI-based (best conditioning in previous tests)
        ('roi_based_ellipsoid', 'ROI-based (centroids)'),
        # Shell-based (NEW)
        ('shell_based_ellipsoid', 'Shell-based (NEW)'),
    ]

    results = []

    for preset_name, description in presets:
        try:
            src_outputs, fwd_outputs = run_pipeline_to_forward(preset_name)
            metrics = compute_forward_conditioning(fwd_outputs, src_outputs)
            metrics['preset'] = preset_name
            metrics['description'] = description
            results.append(metrics)

            print(f"\n  Results for {description}:")
            print(f"    Sources: {metrics['n_sources']}")
            print(f"    Condition number: {metrics['condition_number']:.1f}")
            print(f"    Effective rank (1%): {metrics['effective_rank_1pct']}/{metrics['n_channels']} "
                  f"({100*metrics['rank_ratio']:.1f}%)")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "="*80)
    print("CONDITIONING COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Source Space':<25} {'Sources':>8} {'Cond #':>10} {'Eff Rank':>12} {'Rank %':>8}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x['condition_number']):
        print(f"{r['description']:<25} {r['n_sources']:>8} {r['condition_number']:>10.1f} "
              f"{r['effective_rank_1pct']:>5}/{r['n_channels']:<5} {100*r['rank_ratio']:>7.1f}%")

    print("-"*80)

    # Identify best
    if results:
        best = min(results, key=lambda x: x['condition_number'])
        print(f"\nBest conditioning: {best['description']} (condition # = {best['condition_number']:.1f})")

        # Compare shell-based to others
        shell_result = next((r for r in results if 'shell' in r['preset'].lower()), None)
        if shell_result:
            vol_result = next((r for r in results if 'volumetric' in r['preset'].lower()), None)
            if vol_result:
                improvement = vol_result['condition_number'] / shell_result['condition_number']
                print(f"\nShell-based vs Volumetric:")
                print(f"  Volumetric condition #: {vol_result['condition_number']:.1f}")
                print(f"  Shell-based condition #: {shell_result['condition_number']:.1f}")
                print(f"  Improvement factor: {improvement:.1f}x better conditioning")


if __name__ == '__main__':
    main()
