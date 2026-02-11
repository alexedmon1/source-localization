#!/usr/bin/env python3
"""
Test uniform grid validation with ellipsoid + volumetric source space.

Compares sLORETA, MNE, and dSPM inverse methods using the new uniform
grid test mode to get position-independent localization error metrics.

Features:
- Filters grid to positions with actual sources nearby
- Analyzes localization error by depth from electrodes

Usage:
    python scripts/test_uniform_grid_validation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from scipy.spatial import cKDTree
from source_localization.validation import (
    ValidationRunner, ValidationConfigLoader,
    DipoleSimulator, compute_localization_error,
    compute_roi_classification_accuracy,
    load_atlas_roi_centroids, load_roi_source_mapping,
    generate_uniform_test_grid
)
from source_localization.config import Config
from source_localization.steps import electrode_registration, bem_model, source_space, forward_solution
import json
from datetime import datetime
from collections import Counter
import source_localization


def compute_depth_from_electrodes(position_mm: np.ndarray, electrode_positions_mm: np.ndarray) -> float:
    """Compute minimum distance from a position to any electrode."""
    distances = np.linalg.norm(electrode_positions_mm - position_mm, axis=1)
    return float(np.min(distances))


def run_filtered_grid_validation(config_name: str, n_trials: int = 10, max_dist_to_source: float = 0.7) -> dict:
    """
    Run validation with filtered uniform grid.

    Only tests at grid positions that have a source within max_dist_to_source mm.
    Also computes depth-stratified metrics.
    """
    # Find config
    configs = ValidationConfigLoader.discover_configs('original', config_names=[config_name])
    if not configs:
        raise ValueError(f"Config {config_name} not found")

    config_path = configs[0]
    config = ValidationConfigLoader.load_config(config_path)

    print(f"\n{'='*70}")
    print(f"Running: {config_name} ({config['inverse']['method']})")
    print(f"{'='*70}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'validation_results' / f'{config_name}_filtered_grid'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    # Override config output directory
    config['outputs'] = {'dir': str(output_dir)}

    # Build pipeline components
    print("  Building pipeline...")
    pipeline_config = Config(config)

    # Step 1: Electrodes
    step1 = electrode_registration.run(pipeline_config, {})
    info = step1['info']

    # Get electrode positions
    electrode_positions_mm = np.array([ch['loc'][:3] * 1000 for ch in info['chs']])

    # Step 2: BEM
    step2 = bem_model.run(pipeline_config, step1)

    # Step 3: Source space
    step3 = source_space.run(pipeline_config, {**step1, **step2})
    src = step3['src']

    # Step 4: Forward solution
    step4 = forward_solution.run(pipeline_config, {**step1, **step2, **step3})
    fwd = step4['fwd']

    # Get source positions from forward solution
    source_coords_mm = fwd['source_rr'] * 1000
    n_sources = len(source_coords_mm)
    source_spacing = config['source_space']['volumetric'].get('spacing_mm', 'auto')

    print(f"  Sources: {n_sources} at ~{source_spacing} spacing")
    print(f"  Electrodes: {len(electrode_positions_mm)}")

    # Load atlas for ROI mapping
    package_dir = Path(source_localization.__file__).parent
    roi_labels_file = package_dir / config['inputs']['brain_labels']
    roi_names_file = package_dir / config['inputs'].get('roi_mapping', 'data/atlas/roi_mapping.json')

    # Generate uniform grid
    print(f"\n  Generating uniform grid (1.0mm spacing, 0.2mm margin)...")
    grid_positions, grid_rois = generate_uniform_test_grid(
        str(roi_labels_file),
        spacing_mm=1.0,
        margin_mm=0.2,
        verbose=False
    )
    print(f"  Initial grid: {len(grid_positions)} positions")

    # Filter grid to positions with nearby sources
    print(f"  Filtering to positions with source within {max_dist_to_source}mm...")
    source_tree = cKDTree(source_coords_mm)
    distances, indices = source_tree.query(grid_positions, k=1)

    valid_mask = distances <= max_dist_to_source
    filtered_positions = grid_positions[valid_mask]
    filtered_rois = grid_rois[valid_mask]
    filtered_source_distances = distances[valid_mask]

    print(f"  Filtered grid: {len(filtered_positions)} positions")
    print(f"  Mean distance to nearest source: {np.mean(filtered_source_distances):.3f} mm")

    # Compute depth for each position
    depths = np.array([compute_depth_from_electrodes(pos, electrode_positions_mm)
                       for pos in filtered_positions])
    print(f"  Depth range: {depths.min():.2f} - {depths.max():.2f} mm")

    # Load ROI info
    roi_centroids, roi_names_map = load_atlas_roi_centroids(str(roi_labels_file), str(roi_names_file))
    roi_mapping = load_roi_source_mapping(source_coords_mm, str(roi_labels_file), radius_mm=2.0)

    # Import inverse functions once
    import mne
    from source_localization.steps.inverse_solution import (
        apply_inverse_custom_dSPM,
        apply_inverse_custom_MNE,
        apply_inverse_custom_sLORETA
    )

    # Initialize simulator
    print(f"\n  Running simulations (n_trials={n_trials}, SNR=10dB)...")
    simulator = DipoleSimulator(fwd, info, src, verbose=False)

    # Run validation
    snr_db = 10
    results_per_position = []

    for i, (pos, roi_id, depth) in enumerate(zip(filtered_positions, filtered_rois, depths)):
        if i % 20 == 0:
            print(f"    Position {i+1}/{len(filtered_positions)}...", flush=True)

        for trial in range(n_trials):
            # Simulate dipole
            eeg_data, sim_meta = simulator.simulate_dipole(
                position_mm=pos,
                amplitude_nAm=50.0,
                duration_s=1.0,
                sfreq=500.0,
                snr_db=snr_db,
                noise_seed=trial * 1000 + i,
            )

            # Create Raw and apply inverse
            raw = simulator.create_mne_raw(eeg_data, sfreq=500.0)

            # Apply inverse
            method = config['inverse']['method'].upper()
            snr = config['inverse'].get('snr', 3.0)
            lambda2 = config['inverse'].get('lambda2', 1.0 / snr**2)

            # Create evoked from raw
            events = np.array([[0, 0, 1]])
            epochs = mne.Epochs(raw, events, tmin=0, tmax=raw.times[-1],
                               baseline=None, preload=True, verbose=False)
            evoked = epochs.average()

            # Apply inverse
            if method == 'DSPM':
                source_power = apply_inverse_custom_dSPM(fwd, evoked, snr, lambda2, verbose=False)
            elif method == 'MNE':
                source_power = apply_inverse_custom_MNE(fwd, evoked, snr, lambda2, verbose=False)
            elif method == 'SLORETA':
                source_power = apply_inverse_custom_sLORETA(fwd, evoked, snr, lambda2, verbose=False)
            else:
                raise ValueError(f"Unknown inverse method: {method}")

            # Find peak (source_power has shape [n_sources, n_timepoints])
            # Take mean over time, then find max source
            mean_power = np.mean(np.abs(source_power), axis=1)
            peak_idx = np.argmax(mean_power)
            peak_position = source_coords_mm[peak_idx]

            # Compute metrics
            loc_error = compute_localization_error(sim_meta['actual_position_mm'], peak_position)
            estimated_roi = roi_mapping[peak_idx]
            roi_correct = compute_roi_classification_accuracy(int(roi_id), int(estimated_roi))

            results_per_position.append({
                'position_idx': i,
                'trial': trial,
                'true_position': sim_meta['actual_position_mm'].tolist(),
                'estimated_position': peak_position.tolist(),
                'localization_error': loc_error,
                'true_roi': int(roi_id),
                'estimated_roi': int(estimated_roi),
                'roi_correct': roi_correct,
                'depth': depth,
            })

    # Aggregate results
    errors = [r['localization_error'] for r in results_per_position]
    roi_correct = [r['roi_correct'] for r in results_per_position]
    depths_all = [r['depth'] for r in results_per_position]

    # Depth-stratified analysis
    depth_bins = [(0, 3), (3, 4), (4, 5), (5, 6), (6, 10)]
    depth_results = {}

    for d_min, d_max in depth_bins:
        mask = [(d_min <= d < d_max) for d in depths_all]
        if sum(mask) > 0:
            bin_errors = [e for e, m in zip(errors, mask) if m]
            bin_roi = [r for r, m in zip(roi_correct, mask) if m]
            depth_results[f'{d_min}-{d_max}mm'] = {
                'n_simulations': len(bin_errors),
                'loc_error_mean': float(np.mean(bin_errors)),
                'loc_error_std': float(np.std(bin_errors)),
                'roi_accuracy': float(np.mean(bin_roi)),
            }

    # Summary
    summary = {
        'config_name': config_name,
        'method': config['inverse']['method'],
        'n_sources': n_sources,
        'n_test_positions': len(filtered_positions),
        'n_simulations': len(results_per_position),
        'max_dist_to_source': max_dist_to_source,
        'localization_error': {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
        },
        'roi_accuracy': float(np.mean(roi_correct)),
        'depth_analysis': depth_results,
        'depth_range': {'min': float(min(depths)), 'max': float(max(depths))},
    }

    # Save results
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results:")
    print(f"    Localization error: {summary['localization_error']['mean']:.2f} ± {summary['localization_error']['std']:.2f} mm")
    print(f"    ROI accuracy: {summary['roi_accuracy']*100:.1f}%")
    print(f"\n  Depth analysis:")
    for depth_bin, metrics in depth_results.items():
        print(f"    {depth_bin}: {metrics['loc_error_mean']:.2f}mm error, {metrics['roi_accuracy']*100:.1f}% ROI acc (n={metrics['n_simulations']})")

    return summary


def main():
    print("=" * 70)
    print("FILTERED UNIFORM GRID VALIDATION")
    print("BEM: Ellipsoid | Source Space: Volumetric")
    print("Methods: sLORETA, MNE, dSPM")
    print("Filter: Only positions with source within 0.7mm")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    configs = ['V10', 'V09', 'V08']  # sLORETA, MNE, dSPM
    results_summary = {}

    for config in configs:
        try:
            results = run_filtered_grid_validation(config, n_trials=10, max_dist_to_source=0.7)
            results_summary[config] = results
        except Exception as e:
            print(f"\n✗ {config} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results_summary[config] = {'error': str(e)}

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Filtered Grid, SNR=10dB)")
    print("=" * 70)
    print(f"{'Config':<8} {'Method':<8} {'Loc Error (mm)':<18} {'ROI Acc':<10} {'N Pos':<8}")
    print("-" * 70)

    for config, metrics in results_summary.items():
        if 'error' in metrics:
            print(f"{config:<8} {'ERROR':<8} {metrics['error']}")
        else:
            err = metrics['localization_error']
            err_str = f"{err['mean']:.2f} ± {err['std']:.2f}"
            print(f"{config:<8} {metrics['method']:<8} {err_str:<18} {metrics['roi_accuracy']*100:.1f}%{'':<5} {metrics['n_test_positions']:<8}")

    # Print depth analysis comparison
    print("\n" + "=" * 70)
    print("DEPTH ANALYSIS (Localization Error by Depth)")
    print("=" * 70)

    depth_bins = ['0-3mm', '3-4mm', '4-5mm', '5-6mm', '6-10mm']
    print(f"{'Depth':<10}", end='')
    for config in configs:
        if 'error' not in results_summary[config]:
            print(f"{results_summary[config]['method']:<15}", end='')
    print()
    print("-" * 70)

    for depth_bin in depth_bins:
        print(f"{depth_bin:<10}", end='')
        for config in configs:
            if 'error' not in results_summary[config]:
                depth_data = results_summary[config].get('depth_analysis', {}).get(depth_bin, {})
                if depth_data:
                    print(f"{depth_data['loc_error_mean']:.2f}mm (n={depth_data['n_simulations']:<3})", end=' ')
                else:
                    print(f"{'--':<15}", end='')
        print()

    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save summary
    summary_file = Path(__file__).parent.parent / 'validation_results' / 'filtered_grid_comparison.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
