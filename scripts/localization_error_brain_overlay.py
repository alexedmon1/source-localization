#!/usr/bin/env python3
"""
Localization Error Brain Overlay Figure
========================================

Generates a 3-panel figure (dorsal, sagittal, coronal) showing estimated
localization error as a smooth heatmap overlaid on the skull-stripped
brain anatomy at each view's widest center slice.

Usage:
    # From source-localization repo root:
    uv run python scripts/localization_error_brain_overlay.py

    # Custom pipeline + output:
    uv run python scripts/localization_error_brain_overlay.py \
        --pipeline /path/to/shell/pipeline \
        --output /path/to/output.png

    # Adjust overlay transparency (0-1, lower = more brain visible):
    uv run python scripts/localization_error_brain_overlay.py --alpha 0.5
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, Delaunay


# Validated sLORETA error curve (depth_mm, error_mm) from dipole simulation
SLORETA_ERROR_CURVE = [
    (1.5, 0.46),
    (2.5, 0.74),
    (3.5, 1.71),
    (4.5, 2.67),
    (5.5, 4.31),
]


def make_error_colormap():
    colors = [
        (0.2, 0.8, 0.2),   # Green (low error)
        (0.9, 0.9, 0.1),   # Yellow
        (1.0, 0.65, 0.0),  # Orange
        (1.0, 0.3, 0.0),   # Red-orange
        (0.7, 0.0, 0.0),   # Dark red (high error)
    ]
    return LinearSegmentedColormap.from_list("error", colors, N=512)


def load_atlas(atlas_path):
    """Load skull-stripped brain atlas, correcting the 10x voxel scaling."""
    img = nib.load(atlas_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    affine[:3, :3] *= 0.1
    affine[:3, 3] *= 0.1
    return data, affine


def find_widest_slice(brain_mask, axis):
    """Return the index of the slice with the most nonzero voxels along axis."""
    sum_axes = tuple(i for i in range(3) if i != axis)
    counts = np.sum(brain_mask, axis=sum_axes)
    return int(np.argmax(counts))


def extract_slice_mm(data, affine, axis, slice_idx):
    """Extract a 2D slice and its mm coordinate grids."""
    slicing = [slice(None)] * 3
    slicing[axis] = slice_idx
    slice_2d = data[tuple(slicing)]

    remaining = [i for i in range(3) if i != axis]
    shape = data.shape
    idx0 = np.arange(shape[remaining[0]])
    idx1 = np.arange(shape[remaining[1]])
    I0, I1 = np.meshgrid(idx0, idx1, indexing="ij")

    vox = np.zeros((I0.size, 3))
    vox[:, remaining[0]] = I0.ravel()
    vox[:, remaining[1]] = I1.ravel()
    vox[:, axis] = slice_idx
    mm = nib.affines.apply_affine(affine, vox)

    mm0 = mm[:, remaining[0]].reshape(I0.shape)
    mm1 = mm[:, remaining[1]].reshape(I0.shape)
    return slice_2d, mm0, mm1


def estimate_errors(source_coords_mm, electrode_coords_mm):
    """Estimate localization error at each source from validated depth curve."""
    depths = np.array([p[0] for p in SLORETA_ERROR_CURVE])
    errors = np.array([p[1] for p in SLORETA_ERROR_CURVE])
    dists = np.array([
        np.min(np.linalg.norm(electrode_coords_mm - s, axis=1))
        for s in source_coords_mm
    ])
    return np.interp(dists, depths, errors)


def render_view(
    ax, atlas_slice, mm0, mm1,
    source_coords, errors, electrodes,
    slice_axis, view_name, xlabel, ylabel,
    error_range, cmap, overlay_alpha=0.55,
    grid_res=300, sigma=4.0,
):
    """Render one panel: brain slice background + smooth error overlay."""
    axes_2d = [i for i in range(3) if i != slice_axis]
    coords_2d = source_coords[:, axes_2d]
    elec_2d = electrodes[:, axes_2d]

    # --- Brain anatomy background ---
    brain_norm = atlas_slice / max(atlas_slice.max(), 1e-6)
    brain_norm = gaussian_filter(brain_norm, sigma=0.5)
    ax.pcolormesh(
        mm0, mm1, brain_norm,
        cmap="gray", vmin=0, vmax=1.0,
        shading="gouraud", rasterized=True, zorder=1,
    )

    # Brain outline
    brain_binary = gaussian_filter((atlas_slice > 0.15).astype(float), sigma=1.5)
    ax.contour(
        mm0, mm1, brain_binary, levels=[0.5],
        colors="#222222", linewidths=2.0, alpha=0.9, zorder=10,
    )

    # --- Error heatmap overlay ---
    pad = 0.5
    x_min, x_max = coords_2d[:, 0].min() - pad, coords_2d[:, 0].max() + pad
    y_min, y_max = coords_2d[:, 1].min() - pad, coords_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, grid_res)
    gy = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    rbf = RBFInterpolator(coords_2d, errors, smoothing=2.0, kernel="thin_plate_spline")
    grid_errors = gaussian_filter(rbf(grid_pts).reshape(GX.shape), sigma=sigma)

    # Mask to source hull with soft edges
    hull = ConvexHull(coords_2d)
    inside = Delaunay(coords_2d[hull.vertices]).find_simplex(grid_pts) >= 0
    inside = gaussian_filter(inside.reshape(GX.shape).astype(float), sigma=2.5)
    alpha_map = np.clip(inside * overlay_alpha, 0, overlay_alpha)

    norm = Normalize(vmin=error_range[0], vmax=error_range[1])
    rgba = cmap(norm(np.clip(grid_errors, *error_range)))
    rgba[:, :, 3] = alpha_map

    ax.imshow(
        rgba, extent=[gx[0], gx[-1], gy[0], gy[-1]],
        origin="lower", aspect="auto", interpolation="bilinear", zorder=5,
    )

    # --- Electrodes ---
    ax.scatter(
        elec_2d[:, 0], elec_2d[:, 1],
        c="white", s=55, marker="o",
        edgecolors="black", linewidth=1.5, zorder=100,
    )

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(view_name, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.grid(False)

    margin = 1.5
    ax.set_xlim(coords_2d[:, 0].min() - margin, coords_2d[:, 0].max() + margin)
    ax.set_ylim(coords_2d[:, 1].min() - margin, coords_2d[:, 1].max() + margin)


def generate_figure(
    pipeline_dir,
    atlas_path=None,
    output_path=None,
    overlay_alpha=0.55,
    dpi=200,
):
    """Generate the 3-panel localization error brain overlay figure.

    Parameters
    ----------
    pipeline_dir : Path
        Shell pipeline directory containing data/step3_source_coords_mm.npy
        and data/step1_info.pkl.
    atlas_path : Path, optional
        Skull-stripped brain atlas NIfTI. Defaults to the bundled atlas.
    output_path : Path, optional
        Where to save the PNG. Defaults to figures/ in pipeline_dir.
    overlay_alpha : float
        Max opacity of the error overlay (0-1). Lower = more brain visible.
    dpi : int
        Output resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    pipeline_dir = Path(pipeline_dir)

    if atlas_path is None:
        atlas_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "source_localization" / "data" / "atlas"
            / "Atlas_3DRois_brain.nii.gz"
        )
    atlas_data, affine = load_atlas(atlas_path)
    brain_mask = atlas_data > 0.15

    # Load pipeline data
    source_coords_mm = np.load(pipeline_dir / "data" / "step3_source_coords_mm.npy")
    with open(pipeline_dir / "data" / "step1_info.pkl", "rb") as f:
        info = pickle.load(f)
    electrode_coords_mm = np.array(
        [ch["loc"][:3] * 1000 for ch in info["chs"] if ch["kind"] == 2]
    )

    errors_mm = estimate_errors(source_coords_mm, electrode_coords_mm)
    error_range = (0.3, 4.5)
    cmap = make_error_colormap()

    print(f"Sources: {len(source_coords_mm)}, Electrodes: {len(electrode_coords_mm)}")
    print(f"Error range: {errors_mm.min():.2f} – {errors_mm.max():.2f} mm")

    # Find widest center slices
    views = [
        # (axis, name, xlabel, ylabel)
        (2, "Dorsal", "M-L (mm)", "A-P (mm)"),
        (0, "Sagittal", "A-P (mm)", "D-V (mm)"),
        (1, "Coronal", "M-L (mm)", "D-V (mm)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"right": 0.88})
    fig.patch.set_facecolor("white")

    for ax, (axis, name, xlab, ylab) in zip(axes, views):
        slice_idx = find_widest_slice(brain_mask, axis)
        vox = np.zeros(3)
        vox[axis] = slice_idx
        mm_pos = nib.affines.apply_affine(affine, vox)[axis]
        print(f"  {name}: slice {slice_idx} at {mm_pos:.2f} mm")

        slice_2d, mm0, mm1 = extract_slice_mm(atlas_data, affine, axis, slice_idx)

        render_view(
            ax, slice_2d, mm0, mm1,
            source_coords_mm, errors_mm, electrode_coords_mm,
            slice_axis=axis, view_name=name, xlabel=xlab, ylabel=ylab,
            error_range=error_range, cmap=cmap, overlay_alpha=overlay_alpha,
        )

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(*error_range))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Localization Error (mm)", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(
        "Estimated Localization Error: sLORETA + Ellipsoid BEM",
        fontsize=15, fontweight="bold", y=0.98,
    )
    plt.subplots_adjust(wspace=0.3)

    if output_path is None:
        out_dir = pipeline_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "localization_error_brain_overlay.png"
    output_path = Path(output_path)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("/mnt/d/research/EEG/FORGE/localization/rest_shell/derivatives/sub-801/pipeline"),
        help="Shell pipeline directory",
    )
    parser.add_argument("--atlas", type=Path, default=None, help="Skull-stripped atlas NIfTI")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay opacity (0-1)")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    generate_figure(
        pipeline_dir=args.pipeline,
        atlas_path=args.atlas,
        output_path=args.output,
        overlay_alpha=args.alpha,
        dpi=args.dpi,
    )
