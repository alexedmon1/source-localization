"""Step 8: Visualization.

Generate visualizations of source estimates and ROI activity.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pyvista as pv


def run(config, previous_outputs):
    """
    Generate visualizations.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps containing:
        - 'stc': mne.SourceEstimate - Source time courses
        - 'roi_band_power': dict - ROI band power
        - 'roi_labels': list - ROI names
        - 'primary_band': str - Primary frequency band
        - 'source_coords_mm': ndarray - Source coordinates

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'figures': dict - Dictionary of generated figures
        - 'figure_paths': list - Paths to saved figure files
    """
    output_dir = Path(config['outputs']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_format = config['outputs']['figure_format']
    figure_dpi = config['outputs']['figure_dpi']

    print(f"  Generating visualizations:")
    print(f"    Output directory: {output_dir}")
    print(f"    Format: {figure_format}, DPI: {figure_dpi}")

    # Extract inputs
    stc = previous_outputs['stc']
    roi_band_power = previous_outputs['roi_band_power']
    roi_labels = previous_outputs['roi_labels']
    primary_band = previous_outputs['primary_band']
    source_coords_mm = previous_outputs['source_coords_mm']

    # Fix source count mismatch (MNE filters out some sources)
    n_sources_stc = len(stc.data)
    n_sources_coords = len(source_coords_mm)
    if n_sources_stc != n_sources_coords:
        print(f"    ⚠️  Source count mismatch: stc={n_sources_stc}, coords={n_sources_coords}")
        print(f"    Truncating coordinates to match stc.data")
        source_coords_mm = source_coords_mm[:n_sources_stc]

    figures = {}
    figure_paths = []

    # 1. ROI band power bar plot
    print(f"    Creating ROI {primary_band} power bar plot...")
    fig_roi = plot_roi_band_power(roi_band_power, roi_labels, primary_band)
    figures['roi_band_power'] = fig_roi

    roi_fig_path = output_dir / f'roi_{primary_band}_power.{figure_format}'
    fig_roi.savefig(roi_fig_path, dpi=figure_dpi, bbox_inches='tight')
    figure_paths.append(str(roi_fig_path))
    plt.close(fig_roi)

    # 2. Source activity snapshot (peak time)
    print(f"    Creating source activity snapshot...")
    peak_time_idx = np.abs(stc.data).mean(axis=0).argmax()
    peak_time = stc.times[peak_time_idx]

    fig_source = plot_source_activity_snapshot(
        source_coords_mm,
        stc.data[:, peak_time_idx],
        title=f'Source Activity at t={peak_time*1000:.1f}ms'
    )
    figures['source_snapshot'] = fig_source

    source_fig_path = output_dir / f'source_activity_snapshot.{figure_format}'
    fig_source.savefig(source_fig_path, dpi=figure_dpi, bbox_inches='tight')
    figure_paths.append(str(source_fig_path))
    plt.close(fig_source)

    # 3. Time course plot for top ROI
    print(f"    Creating time course plot...")
    primary_powers = {roi: roi_band_power[roi][primary_band] for roi in roi_labels}
    top_roi = max(primary_powers, key=primary_powers.get)

    fig_timecourse = plot_roi_timecourse(
        previous_outputs['roi_stcs'][top_roi],
        stc.times,
        roi_name=top_roi
    )
    figures['timecourse'] = fig_timecourse

    tc_fig_path = output_dir / f'roi_{top_roi}_timecourse.{figure_format}'
    fig_timecourse.savefig(tc_fig_path, dpi=figure_dpi, bbox_inches='tight')
    figure_paths.append(str(tc_fig_path))
    plt.close(fig_timecourse)

    print(f"    ✓ Generated {len(figure_paths)} figures")

    return {
        'figures': figures,
        'figure_paths': figure_paths
    }


def plot_roi_band_power(roi_band_power, roi_labels, band_name):
    """
    Create bar plot of band power for each ROI.

    Parameters
    ----------
    roi_band_power : dict
        Band power for each ROI
    roi_labels : list
        ROI names
    band_name : str
        Name of frequency band to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Extract band power for specified band
    powers = [roi_band_power[roi][band_name] for roi in roi_labels]

    # Sort by power (descending)
    sorted_indices = np.argsort(powers)[::-1]
    sorted_rois = [roi_labels[i] for i in sorted_indices]
    sorted_powers = [powers[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot top 20 ROIs (or fewer if less than 20)
    n_show = min(20, len(sorted_rois))
    y_pos = np.arange(n_show)

    ax.barh(y_pos, sorted_powers[:n_show])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_rois[:n_show])
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel(f'{band_name.capitalize()} Band Power')
    ax.set_title(f'ROI {band_name.capitalize()} Band Power (Top {n_show})')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_source_activity_snapshot(source_coords_mm, activity, title='Source Activity'):
    """
    Plot source activity as 3D scatter plot.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    activity : ndarray, shape (n_sources,)
        Source activity values
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize activity for color mapping
    activity_abs = np.abs(activity)
    vmax = np.percentile(activity_abs, 95)  # Use 95th percentile for colormap

    # Scatter plot
    scatter = ax.scatter(
        source_coords_mm[:, 0],
        source_coords_mm[:, 1],
        source_coords_mm[:, 2],
        c=activity_abs,
        cmap='hot',
        s=20,
        alpha=0.6,
        vmin=0,
        vmax=vmax
    )

    # Labels and formatting
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Activity (abs)')

    # Equal aspect ratio
    max_range = np.ptp(source_coords_mm, axis=0).max() / 2.0
    mid_x = source_coords_mm[:, 0].mean()
    mid_y = source_coords_mm[:, 1].mean()
    mid_z = source_coords_mm[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig


def plot_roi_timecourse(timecourse, times, roi_name='ROI'):
    """
    Plot time course for a single ROI.

    Parameters
    ----------
    timecourse : ndarray, shape (n_times,)
        ROI time course
    times : ndarray, shape (n_times,)
        Time points in seconds
    roi_name : str
        ROI name for title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Convert times to milliseconds
    times_ms = times * 1000

    ax.plot(times_ms, timecourse, linewidth=1)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Stimulus onset')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Source Activity')
    ax.set_title(f'{roi_name} Time Course')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def create_brain_surface_plot(source_coords_mm, activity, output_file=None):
    """
    Create 3D brain surface plot using PyVista (for interactive visualization).

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
        Source coordinates in mm
    activity : ndarray, shape (n_sources,)
        Source activity values
    output_file : str, optional
        Path to save HTML file for interactive visualization

    Returns
    -------
    plotter : pyvista.Plotter
        PyVista plotter object
    """
    # Create point cloud
    point_cloud = pv.PolyData(source_coords_mm)
    point_cloud['activity'] = np.abs(activity)

    # Create plotter
    plotter = pv.Plotter(off_screen=(output_file is not None))
    plotter.add_points(
        point_cloud,
        scalars='activity',
        cmap='hot',
        point_size=10,
        render_points_as_spheres=True
    )
    plotter.add_axes()
    plotter.camera_position = 'iso'

    if output_file:
        plotter.export_html(output_file)
        plotter.close()
    else:
        plotter.show()

    return plotter
