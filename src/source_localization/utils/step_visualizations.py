"""Visualization functions for each pipeline step."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_step1_electrodes(info, output_path):
    """
    Create electrode registration visualization.

    Parameters
    ----------
    info : mne.Info
        MNE info object with electrode positions
    output_path : Path
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    n_electrodes = len(info['ch_names'])
    elec_coords = np.array([info['chs'][i]['loc'][:3] for i in range(n_electrodes)]) * 1000  # to mm

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Electrode positions (dorsal view)
    ax = axes[0]
    ax.scatter(elec_coords[:, 0], elec_coords[:, 1], c='red', s=100, edgecolors='black', alpha=0.7)
    for i, name in enumerate(info['ch_names']):
        ax.text(elec_coords[i, 0], elec_coords[i, 1], name, fontsize=7, ha='center')
    ax.set_xlabel('X (mm, Medial-Lateral)')
    ax.set_ylabel('Y (mm, Anterior-Posterior)')
    ax.set_title('EEG Electrode Positions on Skull Surface (Dorsal View)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Z-coordinates
    ax = axes[1]
    z_coords = elec_coords[:, 2]
    ax.bar(range(n_electrodes), z_coords, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Electrode')
    ax.set_ylabel('Z (mm, Dorsal-Ventral)')
    ax.set_title('Electrode Z-Coordinates (Dorsal-Ventral Axis)')
    ax.set_xticks(range(n_electrodes))
    ax.set_xticklabels(info['ch_names'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_step3_source_space(source_coords_mm, elec_coords_mm, output_path):
    """
    Create source space visualization with 3 orthogonal views.

    Parameters
    ----------
    source_coords_mm : ndarray
        Source coordinates in mm
    elec_coords_mm : ndarray
        Electrode coordinates in mm
    output_path : Path
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Axial view
    ax = axes[0]
    ax.scatter(source_coords_mm[:, 0], source_coords_mm[:, 1], s=5, c='blue', alpha=0.5)
    ax.scatter(elec_coords_mm[:, 0], elec_coords_mm[:, 1], s=50, c='red', marker='^', edgecolors='black')
    ax.set_xlabel('X (mm, Medial-Lateral)')
    ax.set_ylabel('Y (mm, Anterior-Posterior)')
    ax.set_title('Volumetric Source Space - Axial Plane (Dorsal View)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Coronal view
    ax = axes[1]
    ax.scatter(source_coords_mm[:, 0], source_coords_mm[:, 2], s=5, c='blue', alpha=0.5)
    ax.scatter(elec_coords_mm[:, 0], elec_coords_mm[:, 2], s=50, c='red', marker='^', edgecolors='black')
    ax.set_xlabel('X (mm, Medial-Lateral)')
    ax.set_ylabel('Z (mm, Dorsal-Ventral)')
    ax.set_title('Volumetric Source Space - Coronal Plane (Anterior View)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Sagittal view
    ax = axes[2]
    ax.scatter(source_coords_mm[:, 1], source_coords_mm[:, 2], s=5, c='blue', alpha=0.5)
    ax.scatter(elec_coords_mm[:, 1], elec_coords_mm[:, 2], s=50, c='red', marker='^', edgecolors='black')
    ax.set_xlabel('Y (mm, Anterior-Posterior)')
    ax.set_ylabel('Z (mm, Dorsal-Ventral)')
    ax.set_title('Volumetric Source Space - Sagittal Plane (Lateral View)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_step4_forward(fwd, info, output_path):
    """
    Create forward solution visualization.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    info : mne.Info
        MNE info object
    output_path : Path
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    leadfield = fwd['sol']['data']
    n_electrodes = len(info['ch_names'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Leadfield heatmap (subsampled)
    ax = axes[0]
    subsample = max(1, leadfield.shape[1] // 100)
    im = ax.imshow(leadfield[:, ::subsample], aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(leadfield).max()/2, vmax=np.abs(leadfield).max()/2)
    ax.set_xlabel('Dipole Index (subsampled)')
    ax.set_ylabel('Channel')
    ax.set_title('Leadfield Matrix (Subsampled)')
    ax.set_yticks(range(n_electrodes))
    ax.set_yticklabels(info['ch_names'], fontsize=7)
    plt.colorbar(im, ax=ax, label='Leadfield (V)')

    # Leadfield magnitude per channel
    ax = axes[1]
    leadfield_mag = np.abs(leadfield).mean(axis=1)
    ax.bar(range(n_electrodes), leadfield_mag, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean |Leadfield| (V)')
    ax.set_title('Average Leadfield Sensitivity by EEG Channel')
    ax.set_xticks(range(n_electrodes))
    ax.set_xticklabels(info['ch_names'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_step5_inverse(stc, source_coords_mm, method, output_path, title_suffix=''):
    """
    Create inverse solution visualization.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate
    source_coords_mm : ndarray
        Source coordinates in mm
    method : str
        Inverse method name
    output_path : Path
        Path to save figure
    title_suffix : str
        Optional suffix to add to plot titles (e.g., ' (Magnitude)' or ' (Signed)')

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    # For signed data, use actual values (can be negative)
    # For magnitude data, use absolute values
    source_data = stc.data
    is_signed = np.any(source_data < 0)

    times = stc.times

    if is_signed:
        # For signed data, use RMS (not abs) for summary statistics
        source_rms = np.sqrt(np.mean(source_data**2, axis=1))
    else:
        source_rms = np.sqrt(np.mean(source_data**2, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean source activity over time
    ax = axes[0, 0]
    mean_activity = source_data.mean(axis=0)
    ax.plot(times, mean_activity, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Source Activity (a.u.)')
    ax.set_title(f'Spatially-Averaged Source Activity ({method}){title_suffix}')
    ax.grid(True, alpha=0.3)
    if not is_signed:
        ax.set_ylim(bottom=0)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Top 10 sources by RMS
    ax = axes[0, 1]
    top_idx = np.argsort(source_rms)[-10:]
    for idx in top_idx:
        ax.plot(times, source_data[idx, :], alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Source Activity (a.u.)')
    ax.set_title(f'{method}: Top 10 Active Sources{title_suffix}')
    ax.grid(True, alpha=0.3)
    if not is_signed:
        ax.set_ylim(bottom=0)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Spatial distribution (axial) - use RMS for coloring
    ax = axes[1, 0]
    scatter = ax.scatter(source_coords_mm[:, 0], source_coords_mm[:, 1],
                        c=source_rms, s=10, cmap='hot',
                        vmin=0, vmax=np.percentile(source_rms, 95))
    ax.set_xlabel('X (mm, Medial-Lateral)')
    ax.set_ylabel('Y (mm, Anterior-Posterior)')
    ax.set_title(f'{method} RMS Source Activity - Axial Plane{title_suffix}')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='RMS Activity (a.u.)')

    # Histogram
    ax = axes[1, 1]
    ax.hist(source_rms, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('RMS Source Activity (a.u.)')
    ax.set_ylabel('Count (log scale)')
    ax.set_title(f'Distribution of RMS Source Activity{title_suffix}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_step6_roi_extraction(roi_stcs, roi_labels, stc, roi_source_mapping, output_path,
                                    title_suffix=''):
    """
    Create ROI extraction visualization.

    Parameters
    ----------
    roi_stcs : dict
        ROI time series
    roi_labels : list
        ROI names
    stc : mne.SourceEstimate
        Source estimate for timing info
    roi_source_mapping : dict
        Source indices per ROI
    output_path : Path
        Path to save figure
    title_suffix : str
        Optional suffix to add to plot titles (e.g., ' (Magnitude)' or ' (Signed)')

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    times = stc.times

    # Check if data is signed (has negative values)
    all_values = np.concatenate([ts for ts in roi_stcs.values()])
    is_signed = np.any(all_values < 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ROI time series (first 10)
    ax = axes[0, 0]
    roi_list = sorted(roi_stcs.keys())[:10]
    for roi in roi_list:
        ax.plot(times, roi_stcs[roi], alpha=0.8, linewidth=1.5, label=roi)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ROI Activity (a.u.)')
    ax.set_title(f'Atlas-Based ROI Time Series (First 10 Regions){title_suffix}')
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    if is_signed:
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Mean ROI activity (top 15 by absolute mean for signed, regular mean for magnitude)
    ax = axes[0, 1]
    if is_signed:
        roi_mean_activity = {roi: np.mean(np.abs(ts)) for roi, ts in roi_stcs.items()}
        xlabel = 'Mean |Activity| (a.u.)'
    else:
        roi_mean_activity = {roi: ts.mean() for roi, ts in roi_stcs.items()}
        xlabel = 'Mean Activity (a.u.)'
    sorted_rois = sorted(roi_mean_activity.items(), key=lambda x: x[1], reverse=True)[:15]
    roi_names = [r[0] for r in sorted_rois]
    activities = [r[1] for r in sorted_rois]

    ax.barh(range(len(roi_names)), activities, edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(roi_names)))
    ax.set_yticklabels(roi_names, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(f'Top 15 ROIs by Mean Source Activity{title_suffix}')
    ax.grid(True, alpha=0.3, axis='x')

    # Number of sources per ROI scatter
    ax = axes[1, 0]
    all_roi_names = list(roi_stcs.keys())
    all_activities = [roi_mean_activity[roi] for roi in all_roi_names]
    n_sources_list = [len(roi_source_mapping.get(roi, [])) for roi in all_roi_names]

    scatter = ax.scatter(range(len(all_roi_names)), all_activities, c=n_sources_list, s=100,
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('ROI Index')
    ax.set_ylabel(xlabel)
    ax.set_title(f'ROI Mean Activity vs. Spatial Coverage{title_suffix}')
    plt.colorbar(scatter, ax=ax, label='# Sources')
    ax.grid(True, alpha=0.3)

    # ROI Heatmap (top 30 ROIs × time)
    ax = axes[1, 1]
    n_display = min(30, len(roi_stcs))
    sorted_rois_all = sorted(roi_mean_activity.items(), key=lambda x: x[1], reverse=True)[:n_display]
    heatmap_data = np.array([roi_stcs[r[0]] for r in sorted_rois_all])

    # Subsample time for visualization
    time_subsample = max(1, heatmap_data.shape[1] // 100)
    heatmap_data_sub = heatmap_data[:, ::time_subsample]
    times_sub = times[::time_subsample]

    # Use different colormap for signed vs magnitude data
    if is_signed:
        vmax = np.percentile(np.abs(heatmap_data_sub), 95)
        im = ax.imshow(heatmap_data_sub, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                      vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(heatmap_data_sub, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ROI')
    ax.set_title(f'Regional Source Activity Heatmap (Top {n_display} ROIs){title_suffix}')
    ax.set_yticks(range(0, n_display, max(1, n_display//10)))
    ax.set_yticklabels([sorted_rois_all[i][0] for i in range(0, n_display, max(1, n_display//10))], fontsize=7)

    # Set x-ticks to show time
    n_xticks = min(10, len(times_sub))
    xtick_idx = np.linspace(0, len(times_sub)-1, n_xticks, dtype=int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([f'{times_sub[i]:.2f}' for i in xtick_idx], fontsize=7)
    plt.colorbar(im, ax=ax, label='Power')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_step7_band_power(roi_band_power, freq_bands, output_path):
    """
    Create band power visualization.

    Parameters
    ----------
    roi_band_power : dict
        Band power for each ROI
    freq_bands : dict
        Frequency band definitions
    output_path : Path
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        Generated figure
    """
    # Filter out ROIs with no sources (zero power across all bands)
    roi_names_all = sorted(roi_band_power.keys())
    roi_names = []
    for roi in roi_names_all:
        total_power = sum(roi_band_power[roi].values())
        if total_power > 1e-30:  # Keep ROIs with any measurable power
            roi_names.append(roi)

    n_bands = len(freq_bands)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (band_name, band_range) in enumerate(freq_bands.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        # roi_band_power is organized as: roi_band_power[roi_name][band_name] = power
        # Need to extract power for this band across filtered ROIs
        powers = [roi_band_power[roi][band_name] for roi in roi_names]

        ax.bar(range(len(roi_names)), powers, edgecolor='black', alpha=0.7)
        ax.set_xlabel('ROI')
        ax.set_ylabel(f'{band_name.capitalize()} Power (a.u.)')
        ax.set_title(f'{band_name.capitalize()} Band Activity ({band_range[0]}-{band_range[1]} Hz)')
        ax.set_xticks(range(0, len(roi_names), max(1, len(roi_names)//10)))
        ax.set_xticklabels([str(i) for i in range(0, len(roi_names), max(1, len(roi_names)//10))], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide extra subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
