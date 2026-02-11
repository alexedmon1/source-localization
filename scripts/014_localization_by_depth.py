#!/usr/bin/env python
"""
Localization error by depth/distance from electrodes.

Tests dipoles at different depths and measures how localization
accuracy degrades with depth. EEG has better resolution for
superficial sources than deep sources.

Usage:
    python scripts/014_localization_by_depth.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import numpy as np
import mne
from source_localization.config import Config
from source_localization.steps import electrode_registration, bem_model, source_space, forward_solution
from source_localization.validation.simulation import DipoleSimulator
from source_localization.inverse import methods as inverse_methods
from source_localization.utils.shell_mapping import get_shell_metadata


def compute_distance_to_electrodes(position_mm, electrode_positions_mm):
    """Compute minimum distance from position to any electrode."""
    distances = np.linalg.norm(electrode_positions_mm - position_mm, axis=1)
    return np.min(distances)


def run_depth_analysis(preset_name, n_per_depth=20, snr_db=10.0):
    """Run localization test stratified by depth."""
    print(f"\n{'='*60}")
    print(f"DEPTH ANALYSIS: {preset_name}")
    print(f"{'='*60}")

    # Setup pipeline
    config = Config.from_preset(preset_name)

    if 'shell_based' in preset_name:
        config['source_space']['shell_based'].update({
            'n_shells': 4,
            'shell_scales': [0.3, 0.5, 0.7, 0.9],
            'min_points_per_shell': 40,
            'max_points_per_shell': 160,
            'scale_by_area': True,
        })

    config['outputs']['save_intermediate'] = False
    config['outputs']['dir'] = '/tmp/depth_analysis'

    # Run pipeline
    elec_outputs = electrode_registration.run(config, {})
    bem_outputs = bem_model.run(config, elec_outputs)
    previous = {**elec_outputs, **bem_outputs}
    src_outputs = source_space.run(config, previous)
    previous.update(src_outputs)
    fwd_outputs = forward_solution.run(config, previous)

    source_coords_mm = src_outputs['source_coords_mm']
    n_sources = src_outputs['n_sources']
    info = elec_outputs['info']
    fwd = fwd_outputs['fwd']
    src = src_outputs['src']
    bem_params = bem_outputs['bem_params']

    # Get electrode positions
    electrode_positions_mm = np.array([
        ch['loc'][:3] * 1000 for ch in info['chs']
    ])

    print(f"\n  Sources: {n_sources}")
    print(f"  Electrodes: {len(electrode_positions_mm)}")

    # Get depth metadata for all sources
    shell_scales = config['source_space'].get('shell_based', {}).get('shell_scales', None)
    depth_meta = get_shell_metadata(source_coords_mm, bem_params, shell_scales)

    # Initialize simulator
    simulator = DipoleSimulator(fwd, info, src, verbose=False)

    # Sample sources at different depths
    # Bin by normalized depth: [0-0.4], [0.4-0.6], [0.6-0.8], [0.8-1.0]
    depth_bins = [(0.0, 0.4, 'deep'), (0.4, 0.6, 'mid-deep'),
                  (0.6, 0.8, 'mid-superficial'), (0.8, 1.0, 'superficial')]

    results = []

    for d_min, d_max, depth_label in depth_bins:
        # Find sources in this depth range
        mask = (depth_meta['normalized_depth'] >= d_min) & (depth_meta['normalized_depth'] < d_max)
        if d_max == 1.0:  # Include upper bound for last bin
            mask = (depth_meta['normalized_depth'] >= d_min) & (depth_meta['normalized_depth'] <= d_max)

        depth_sources = np.where(mask)[0]

        if len(depth_sources) < n_per_depth:
            print(f"\n  {depth_label} ({d_min:.1f}-{d_max:.1f}): only {len(depth_sources)} sources, using all")
            test_sources = depth_sources
        else:
            np.random.seed(42)
            test_sources = np.random.choice(depth_sources, n_per_depth, replace=False)

        print(f"\n  Testing {depth_label} ({d_min:.1f}-{d_max:.1f}): {len(test_sources)} dipoles")

        for i, src_idx in enumerate(test_sources):
            test_position = source_coords_mm[src_idx]
            normalized_depth = depth_meta['normalized_depth'][src_idx]

            # Distance to nearest electrode
            dist_to_electrodes = compute_distance_to_electrodes(test_position, electrode_positions_mm)

            # Simulate dipole
            eeg_data, sim_meta = simulator.simulate_dipole(
                position_mm=test_position,
                amplitude_nAm=50.0,
                snr_db=snr_db,
                duration_s=0.5,
                sfreq=500.0,
                noise_seed=42 + i
            )

            # Apply inverse
            evoked = mne.EvokedArray(eeg_data, info, tmin=0, verbose=False)
            source_activity_norm, _, _ = inverse_methods.apply_inverse_sLORETA(
                fwd, info, evoked=evoked, snr=3.0
            )

            # Find peak
            n_sources_inv = source_activity_norm.shape[0] // 3
            source_reshaped = source_activity_norm.reshape(n_sources_inv, 3, -1)
            source_magnitude = np.linalg.norm(source_reshaped, axis=1)
            peak_idx = np.argmax(source_magnitude.mean(axis=1))
            peak_position_mm = simulator.source_positions[peak_idx]

            # Localization error
            localization_error = np.linalg.norm(test_position - peak_position_mm)

            results.append({
                'depth_label': depth_label,
                'normalized_depth': normalized_depth,
                'dist_to_electrodes_mm': dist_to_electrodes,
                'true_position_mm': test_position,
                'peak_position_mm': peak_position_mm,
                'localization_error_mm': localization_error,
            })

        # Per-depth summary
        depth_results = [r for r in results if r['depth_label'] == depth_label]
        errors = [r['localization_error_mm'] for r in depth_results]
        dists = [r['dist_to_electrodes_mm'] for r in depth_results]
        print(f"    Electrode distance: {np.mean(dists):.1f} ± {np.std(dists):.1f} mm")
        print(f"    Localization error: {np.mean(errors):.2f} ± {np.std(errors):.2f} mm")

    return results


def main():
    print("="*70)
    print("LOCALIZATION ERROR BY DEPTH / ELECTRODE DISTANCE")
    print("="*70)

    # Test shell-based
    results_shell = run_depth_analysis('shell_based_ellipsoid', n_per_depth=15, snr_db=10.0)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: shell_based_ellipsoid")
    print(f"{'='*70}")
    print(f"{'Depth':<18} {'Norm Depth':>12} {'Elec Dist':>12} {'Loc Error':>12}")
    print("-"*56)

    depth_labels = ['deep', 'mid-deep', 'mid-superficial', 'superficial']
    for label in depth_labels:
        dr = [r for r in results_shell if r['depth_label'] == label]
        if dr:
            mean_nd = np.mean([r['normalized_depth'] for r in dr])
            mean_ed = np.mean([r['dist_to_electrodes_mm'] for r in dr])
            mean_le = np.mean([r['localization_error_mm'] for r in dr])
            std_le = np.std([r['localization_error_mm'] for r in dr])
            print(f"{label:<18} {mean_nd:>10.2f}   {mean_ed:>10.1f} mm {mean_le:>8.2f} ± {std_le:.2f} mm")

    print("-"*56)

    # Overall correlation
    all_depths = [r['normalized_depth'] for r in results_shell]
    all_dists = [r['dist_to_electrodes_mm'] for r in results_shell]
    all_errors = [r['localization_error_mm'] for r in results_shell]

    from scipy.stats import pearsonr, spearmanr

    r_depth, p_depth = pearsonr(all_depths, all_errors)
    r_dist, p_dist = pearsonr(all_dists, all_errors)
    rho_depth, _ = spearmanr(all_depths, all_errors)
    rho_dist, _ = spearmanr(all_dists, all_errors)

    print(f"\nCorrelations with localization error:")
    print(f"  Normalized depth:    r={r_depth:.3f} (p={p_depth:.4f}), rho={rho_depth:.3f}")
    print(f"  Electrode distance:  r={r_dist:.3f} (p={p_dist:.4f}), rho={rho_dist:.3f}")

    # Interpretation
    print(f"\nInterpretation:")
    if r_depth < -0.2:
        print(f"  - Deeper sources have BETTER localization (unexpected)")
    elif r_depth > 0.2:
        print(f"  - Deeper sources have WORSE localization (expected depth bias)")
    else:
        print(f"  - No strong depth bias detected")

    if r_dist > 0.3:
        print(f"  - Sources farther from electrodes have worse localization")


if __name__ == '__main__':
    main()
