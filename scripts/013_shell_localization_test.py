#!/usr/bin/env python
"""
Quick localization accuracy test for shell-based source space.

Simulates dipoles at random brain locations and measures:
1. Snapping error (how far from requested position to nearest source)
2. Localization error (how far from estimated peak to ground truth)
3. ROI classification accuracy

Usage:
    python scripts/013_shell_localization_test.py

This is a quick test (5 ROIs, 10 trials each = 50 simulations).
For full validation, use the validation CLI.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

import numpy as np
import json
import nibabel as nib
from source_localization.config import Config
from source_localization.steps import (
    electrode_registration, bem_model, source_space, forward_solution
)
from source_localization.validation.simulation import DipoleSimulator
from source_localization.inverse import methods as inverse_methods
from source_localization.utils.shell_mapping import assign_sources_to_rois
import mne


def run_localization_test(
    preset_name,
    n_rois=5,
    n_trials=10,
    snr_db=10.0
):
    """Run localization test for a given preset."""
    print(f"\n{'='*60}")
    print(f"LOCALIZATION TEST: {preset_name}")
    print(f"{'='*60}")

    # Setup pipeline
    config = Config.from_preset(preset_name)

    # Use 4-shell-dense for shell-based
    if 'shell_based' in preset_name:
        config['source_space']['shell_based'].update({
            'n_shells': 4,
            'shell_scales': [0.3, 0.5, 0.7, 0.9],
            'min_points_per_shell': 40,
            'max_points_per_shell': 160,
            'scale_by_area': True,
        })

    config['outputs']['save_intermediate'] = False
    config['outputs']['dir'] = '/tmp/localization_test'

    # Run pipeline through forward solution
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

    print(f"\n  Sources: {n_sources}")

    # Load atlas for ROI centroids
    package_dir = Path(__file__).parent.parent / "src/source_localization"
    labels_path = package_dir / "data/atlas/Atlas_3DRoisLeftRight.Labels.nii"
    roi_mapping_path = package_dir / "data/atlas/roi_mapping.json"

    nii_labels = nib.load(labels_path)
    label_data = nii_labels.get_fdata()

    with open(roi_mapping_path) as f:
        roi_data = json.load(f)
    roi_mapping = roi_data.get('rois', roi_data)

    # Get ROI centroids
    from source_localization.utils.atlas import get_true_affine
    affine = get_true_affine(nii_labels)

    unique_labels = np.unique(label_data[label_data > 0]).astype(int)
    roi_centroids = {}
    for label_id in unique_labels:
        if str(label_id) not in roi_mapping:
            continue
        mask = label_data == label_id
        voxels = np.argwhere(mask)
        centroid_vox = voxels.mean(axis=0)
        centroid_mm = nib.affines.apply_affine(affine, centroid_vox)
        roi_name = roi_mapping[str(label_id)].get('name', f'ROI_{label_id}')
        roi_centroids[label_id] = {
            'name': roi_name,
            'centroid_mm': centroid_mm
        }

    # Select random ROIs for testing
    available_rois = list(roi_centroids.keys())
    np.random.seed(42)
    test_rois = np.random.choice(available_rois, min(n_rois, len(available_rois)), replace=False)

    print(f"  Testing {len(test_rois)} ROIs, {n_trials} trials each")

    # Initialize simulator
    simulator = DipoleSimulator(fwd, info, src, verbose=False)

    # Results storage
    results = []

    for roi_id in test_rois:
        roi_info = roi_centroids[roi_id]
        roi_name = roi_info['name']
        centroid_mm = roi_info['centroid_mm']

        print(f"\n  ROI: {roi_name} (centroid: [{centroid_mm[0]:.1f}, {centroid_mm[1]:.1f}, {centroid_mm[2]:.1f}] mm)")

        for trial in range(n_trials):
            # Add small jitter to centroid
            jitter = np.random.randn(3) * 0.5  # 0.5mm std
            test_position = centroid_mm + jitter

            # Simulate dipole
            eeg_data, sim_meta = simulator.simulate_dipole(
                position_mm=test_position,
                amplitude_nAm=50.0,
                snr_db=snr_db,
                duration_s=0.5,
                sfreq=500.0,
                noise_seed=42 + trial
            )

            snapping_error = sim_meta['snapping_error_mm']

            # Create MNE Raw object for inverse
            raw = mne.io.RawArray(eeg_data, info, verbose=False)
            raw.set_eeg_reference(projection=True, verbose=False)

            # Apply inverse using sLORETA for best localization
            evoked = mne.EvokedArray(eeg_data, info, tmin=0, verbose=False)

            source_activity_norm, source_signed, W = inverse_methods.apply_inverse_sLORETA(
                fwd, info,
                evoked=evoked,
                snr=3.0
            )

            # Find peak: reshape to (n_sources, 3, n_times), compute norm across orientations
            n_sources_inv = source_activity_norm.shape[0] // 3
            source_reshaped = source_activity_norm.reshape(n_sources_inv, 3, -1)
            source_magnitude = np.linalg.norm(source_reshaped, axis=1)  # (n_sources, n_times)
            peak_idx = np.argmax(source_magnitude.mean(axis=1))

            # Get peak position from forward solution source positions
            peak_position_mm = simulator.source_positions[peak_idx]

            # Compute localization error (from ground truth, not snapped)
            localization_error = np.linalg.norm(test_position - peak_position_mm)

            # Get ROI of estimated peak
            peak_roi_ids, _ = assign_sources_to_rois(
                peak_position_mm.reshape(1, -1),
                nii_labels,
                roi_mapping
            )
            estimated_roi_id = peak_roi_ids[0]

            # ROI correct?
            roi_correct = (estimated_roi_id == roi_id)

            results.append({
                'roi_id': roi_id,
                'roi_name': roi_name,
                'true_position_mm': test_position,
                'peak_position_mm': peak_position_mm,
                'snapping_error_mm': snapping_error,
                'localization_error_mm': localization_error,
                'estimated_roi_id': estimated_roi_id,
                'roi_correct': roi_correct,
            })

        # Print per-ROI summary
        roi_results = [r for r in results if r['roi_id'] == roi_id]
        mean_loc_error = np.mean([r['localization_error_mm'] for r in roi_results])
        mean_snap_error = np.mean([r['snapping_error_mm'] for r in roi_results])
        roi_accuracy = np.mean([r['roi_correct'] for r in roi_results])
        print(f"    Snapping error: {mean_snap_error:.2f} mm")
        print(f"    Localization error: {mean_loc_error:.2f} mm")
        print(f"    ROI accuracy: {100*roi_accuracy:.0f}%")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {preset_name}")
    print(f"{'='*60}")

    all_loc_errors = [r['localization_error_mm'] for r in results]
    all_snap_errors = [r['snapping_error_mm'] for r in results]
    all_roi_correct = [r['roi_correct'] for r in results]

    print(f"  Total simulations: {len(results)}")
    print(f"  Snapping error: mean={np.mean(all_snap_errors):.2f} mm, "
          f"median={np.median(all_snap_errors):.2f} mm")
    print(f"  Localization error: mean={np.mean(all_loc_errors):.2f} mm, "
          f"median={np.median(all_loc_errors):.2f} mm, "
          f"std={np.std(all_loc_errors):.2f} mm")
    print(f"  ROI accuracy: {100*np.mean(all_roi_correct):.1f}%")

    return results


def main():
    print("="*70)
    print("SHELL-BASED LOCALIZATION ACCURACY TEST")
    print("="*70)

    # Test configurations
    configs = [
        ('shell_based_ellipsoid', 'Shell-based (4-shell-dense)'),
        ('roi_based_ellipsoid', 'ROI-based (baseline)'),
    ]

    all_results = {}

    for preset_name, description in configs:
        try:
            results = run_localization_test(
                preset_name,
                n_rois=5,
                n_trials=10,
                snr_db=10.0
            )
            all_results[preset_name] = results
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Configuration':<30} {'Loc Error':>12} {'ROI Acc':>10}")
        print("-"*55)

        for preset_name, results in all_results.items():
            mean_error = np.mean([r['localization_error_mm'] for r in results])
            roi_acc = np.mean([r['roi_correct'] for r in results])
            print(f"{preset_name:<30} {mean_error:>10.2f} mm {100*roi_acc:>9.1f}%")


if __name__ == '__main__':
    main()
