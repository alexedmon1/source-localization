"""
3D volume rendering of where the inverse solution thinks a source is.

The resolution metrics (PLE, spatial dispersion, resolution distance) compress a
reconstruction to a number. This renders the thing itself: the reconstructed
activation as a translucent cloud in brain space, with the true dipole positions
marked, so the blur and the merging of nearby sources are visible directly.

The cloud is the reconstructed activation normalized to its peak — a relative
plausibility map, *not* a calibrated posterior probability. Two sources close
together show up as the characteristic picture: two fuzzy balls joined by a
low-intensity bridge.

Functions
---------
interpolate_to_grid
    Scattered source values -> a regular 3D grid suitable for volume rendering.
render_activation_cloud
    Volume-render an activation map with true-source markers.
"""
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ['interpolate_to_grid', 'render_activation_cloud', 'sample_on_plane']

#: Opacity transfer function. Deliberately convex: high activation renders
#: nearly solid while the low tail stays faint but visible, which is what makes
#: the bridge between two nearby sources readable instead of either invisible or
#: a solid blob.
DEFAULT_OPACITY = [0.0, 0.012, 0.04, 0.10, 0.22, 0.45, 0.80]


def interpolate_to_grid(
    positions_mm: np.ndarray,
    values: np.ndarray,
    resolution: int = 64,
    padding_mm: float = 1.5,
    sigma_mm: Optional[float] = None,
    truncate: float = 3.0,
):
    """
    Resample scattered per-source values onto a regular grid by Gaussian splatting.

    Each source contributes ``v_i * exp(-d^2 / 2 sigma^2)`` to nearby grid points
    and the contributions are summed — a blur of the source cloud. This is chosen
    over an RBF interpolant on purpose: a thin-plate spline is a *global*,
    unbounded kernel and rings badly on this irregular grid, painting periodic
    stripes across the volume that look like structure but are pure artifact.
    Splatting is strictly local, cannot ring, cannot overshoot outside the data
    range, and decays to zero away from the sources, which is also exactly the
    soft-edged look the render wants.

    Parameters
    ----------
    positions_mm : ndarray, shape (n, 3)
        Source positions.
    values : ndarray, shape (n,)
        Value per source (e.g. reconstructed activation). Clipped at 0.
    resolution : int, default=64
        Grid points along the longest axis.
    padding_mm : float, default=1.5
        Margin added around the source bounding box.
    sigma_mm : float, optional
        Splat width. Defaults to the median nearest-neighbour source spacing,
        which fills the gaps between sources without blurring away real
        structure.
    truncate : float, default=3.0
        Splat radius in units of sigma.

    Returns
    -------
    grid : pyvista.ImageData
        Grid carrying the field in point array ``activation``, normalized so the
        peak is 1.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    pos = np.asarray(positions_mm, float)
    vals = np.clip(np.asarray(values, float), 0, None)

    if sigma_mm is None:
        d, _ = cKDTree(pos).query(pos, k=2)
        sigma_mm = float(np.median(d[:, 1]))

    lo = pos.min(axis=0) - padding_mm
    hi = pos.max(axis=0) + padding_mm
    extent = hi - lo

    spacing = extent.max() / (resolution - 1)
    dims = np.maximum((extent / spacing).astype(int) + 1, 2).astype(int)

    axes = [lo[i] + np.arange(dims[i]) * spacing for i in range(3)]
    field = np.zeros(tuple(dims), dtype=float)

    weight = np.zeros(tuple(dims), dtype=float)

    reach = truncate * sigma_mm
    two_sigma_sq = 2.0 * sigma_mm ** 2
    for p, v in zip(pos, vals):
        # Only touch the sub-box within `reach` of this source. Every source
        # contributes to `weight` even when its value is 0, since `weight` is the
        # local sampling density.
        sl, offs = [], []
        for i in range(3):
            i0 = max(int(np.floor((p[i] - reach - lo[i]) / spacing)), 0)
            i1 = min(int(np.ceil((p[i] + reach - lo[i]) / spacing)) + 1, dims[i])
            if i0 >= i1:
                break
            sl.append(slice(i0, i1))
            offs.append(axes[i][i0:i1] - p[i])
        if len(sl) < 3:
            continue
        d2 = (offs[0][:, None, None] ** 2 + offs[1][None, :, None] ** 2
              + offs[2][None, None, :] ** 2)
        w = np.exp(-d2 / two_sigma_sq)
        weight[tuple(sl)] += w
        field[tuple(sl)] += v * w

    # NOTE: the field is the raw weighted SUM, which is density-weighted — where
    # sources sit closer together the cloud brightens slightly, independent of
    # activation. Dividing by `weight` removes that bias but costs more than it
    # buys: at the sparse rim the divisor is tiny, so the edge is amplified into
    # a bright shell and the interior structure washes out (tried both ways;
    # normalizing made the two-source lobes markedly harder to see). The real
    # source grid is near-uniform (median spacing ~1.25 mm), so the bias is small
    # — on it the cloud peak lands on the true source. Keep this in mind before
    # reading brightness quantitatively on a strongly non-uniform source space.

    peak = field.max()
    if peak > 0:
        field /= peak

    grid = pv.ImageData(dimensions=tuple(int(d) for d in dims),
                        spacing=(spacing,) * 3, origin=tuple(lo))
    # VTK point arrays are Fortran-ordered relative to (nx, ny, nz) indexing.
    grid.point_data['activation'] = field.ravel(order='F')
    return grid


def sample_on_plane(
    positions_mm: np.ndarray,
    values: np.ndarray,
    center_mm: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    half_extent_mm: float = 8.0,
    n: int = 220,
    sigma_mm: Optional[float] = None,
    reference_peak: Optional[float] = None,
):
    """
    Sample the activation field on an arbitrary plane, for 2D cross-sections.

    Kernel-weighted **average** (Gaussian Nadaraya-Watson), not the weighted sum
    used by :func:`interpolate_to_grid`. The difference matters whenever the
    slice is used to *verify* a threshold rather than just look good: a sum is
    density-weighted, so a cluster of moderate sources can out-total a single
    sharp peak. Sampling the real data that way put the plane's maximum on a
    dense patch and rendered the map's actual global peak at 0.40 of it, which
    made drawn contours disagree with the thresholds the metrics use. An average
    reproduces each source's own value at its own location, so a contour at 50%
    really is the 50% contour.

    Pass ``reference_peak`` (normally ``activation.max()``, the same peak the
    separability metrics normalize by) so the returned field is in units of
    "fraction of map peak" and the drawn contour matches the metric exactly.
    Without it the field is normalized to its own in-plane maximum, which is only
    correct when the plane happens to contain the global peak.

    Parameters
    ----------
    positions_mm : ndarray, shape (n_src, 3)
    values : ndarray, shape (n_src,)
    center_mm : ndarray, shape (3,)
        Plane origin (e.g. the midpoint of two sources).
    e1, e2 : ndarray, shape (3,)
        In-plane basis vectors; orthonormalized here.
    half_extent_mm : float
        Half-width of the sampled square.
    n : int
        Samples per side.
    sigma_mm : float, optional
        Splat width; defaults to median nearest-neighbour source spacing.

    Returns
    -------
    field : ndarray, shape (n, n)
        Activation on the plane, normalized to its own peak.
    coords : ndarray, shape (n,)
        Axis coordinates in mm, relative to ``center_mm`` (same for both axes).
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(positions_mm, float)
    vals = np.clip(np.asarray(values, float), 0, None)

    if sigma_mm is None:
        d, _ = cKDTree(pos).query(pos, k=2)
        sigma_mm = float(np.median(d[:, 1]))

    e1 = np.asarray(e1, float)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.asarray(e2, float)
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    t = np.linspace(-half_extent_mm, half_extent_mm, n)
    uu, vv = np.meshgrid(t, t, indexing='xy')
    pts = (np.asarray(center_mm, float)[None, :]
           + uu.ravel()[:, None] * e1[None, :]
           + vv.ravel()[:, None] * e2[None, :])

    # Gaussian-weighted average over sources within 3 sigma.
    field = np.zeros(pts.shape[0])
    tree = cKDTree(pos)
    neighbours = tree.query_ball_point(pts, r=3.0 * sigma_mm)
    for k, idx in enumerate(neighbours):
        if not idx:
            continue
        idx = np.asarray(idx)
        d2 = np.sum((pos[idx] - pts[k]) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * sigma_mm ** 2))
        wsum = w.sum()
        if wsum > 0:
            field[k] = float(np.dot(vals[idx], w) / wsum)

    field = field.reshape(n, n)
    peak = reference_peak if reference_peak is not None else field.max()
    if peak and peak > 0:
        field = field / peak
    return field, t


def render_activation_cloud(
    positions_mm: np.ndarray,
    activation: np.ndarray,
    out_path,
    true_positions_mm: Optional[Sequence[np.ndarray]] = None,
    peak_position_mm: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    cmap: str = 'inferno',
    resolution: int = 64,
    window_size: Tuple[int, int] = (1900, 760),
    views: Sequence[str] = ('iso', 'top', 'side'),
    show_outline: bool = True,
    zoom: float = 1.5,
) -> Path:
    """
    Volume-render an activation map, marking the true source location(s).

    Parameters
    ----------
    positions_mm : ndarray, shape (n, 3)
        Source positions.
    activation : ndarray, shape (n,)
        Reconstructed activation per source.
    out_path : path-like
        PNG to write.
    true_positions_mm : sequence of ndarray, optional
        Ground-truth dipole positions, drawn as ringed markers.
    peak_position_mm : ndarray, optional
        Reconstruction peak, drawn as a cross — the gap between this and the
        true marker is the localization error made visible.
    title : str, optional
        Rendered into the top-left of the image.
    cmap : str, default='inferno'
        Colormap for the cloud.
    resolution : int, default=64
        Interpolation grid resolution.
    window_size : tuple, default=(1900, 760)
        Output image size in pixels. Wide and short suits a row of views.
    views : sequence of str
        Any of ``'iso'``, ``'top'``, ``'side'``, ``'front'``. One subplot each.
    show_outline : bool, default=True
        Draw a faint hull around the source space for anatomical context.
    zoom : float, default=1.25
        Camera zoom applied to every view.

    Returns
    -------
    Path
        The written file.
    """
    import pyvista as pv

    pos = np.asarray(positions_mm, float)
    grid = interpolate_to_grid(pos, activation, resolution=resolution)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size),
                         shape=(1, len(views)), border=False)
    plotter.set_background('white')

    hull = None
    if show_outline:
        hull = pv.PolyData(pos).delaunay_3d().extract_surface(
            algorithm='dataset_surface')

    for col, view in enumerate(views):
        plotter.subplot(0, col)

        plotter.add_volume(
            grid, scalars='activation', cmap=cmap, opacity=DEFAULT_OPACITY,
            clim=[0.0, 1.0], shade=False,
            show_scalar_bar=(col == 0),   # one shared colorbar is enough
            scalar_bar_args={'title': 'normalized activation',
                             'color': '#222222', 'title_font_size': 16,
                             'label_font_size': 13, 'n_labels': 3,
                             'vertical': False, 'width': 0.44,
                             'height': 0.07,
                             'position_x': 0.03, 'position_y': 0.03},
        )

        if hull is not None:
            plotter.add_mesh(hull, color='#3a6ea5', opacity=0.055,
                             smooth_shading=True, show_scalar_bar=False)

        for p in (true_positions_mm or []):
            p = np.asarray(p, float)
            # Small solid core inside a translucent shell: stays visible where
            # the cloud is bright without hiding the activation around it.
            plotter.add_mesh(pv.Sphere(radius=0.28, center=p), color='#00E5FF',
                             show_scalar_bar=False)
            plotter.add_mesh(pv.Sphere(radius=0.75, center=p), color='#00E5FF',
                             opacity=0.22, show_scalar_bar=False)

        if peak_position_mm is not None:
            pk = np.asarray(peak_position_mm, float)
            for axis in np.eye(3):
                plotter.add_mesh(
                    pv.Line(pk - 0.85 * axis, pk + 0.85 * axis),
                    color='#39FF14', line_width=4, show_scalar_bar=False)

        _set_view(plotter, view, true_positions_mm)
        plotter.camera.zoom(zoom)
        plotter.add_text(view, position='lower_left', font_size=10,
                         color='#666666')

    if title:
        plotter.subplot(0, 0)
        plotter.add_text(title, position='upper_left', font_size=12,
                         color='#111111')

    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path


def _set_view(plotter, view: str, true_positions=None) -> None:
    """Point the camera. Coordinates are (X=L-R, Y=A-P, Z=D-V, array dorsal)."""
    if view == 'top':          # looking down from the electrode array
        plotter.view_xy()
    elif view == 'side':       # sagittal, shows depth below the array
        plotter.view_yz()
    elif view == 'front':      # coronal
        plotter.view_xz()
    elif view == 'pair':
        _view_across_pair(plotter, true_positions)
    else:                      # iso
        plotter.view_isometric()


def _view_across_pair(plotter, true_positions) -> None:
    """
    Look perpendicular to the line joining two sources.

    Volume rendering integrates along the view ray, so a pair that happens to lie
    near the viewing axis projects on top of itself and reads as one blob no
    matter how well separated it is. This puts the separation axis across the
    image, which is the only fair way to look at two-source structure.
    """
    if true_positions is None or len(true_positions) < 2:
        plotter.view_isometric()
        return

    p1, p2 = np.asarray(true_positions[0], float), np.asarray(true_positions[1], float)
    axis = p2 - p1
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        plotter.view_isometric()
        return
    axis /= norm

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, up))) > 0.95:      # pair runs vertically
        up = np.array([0.0, 1.0, 0.0])
    direction = np.cross(axis, up)
    direction /= np.linalg.norm(direction)

    focus = 0.5 * (p1 + p2)
    distance = max(6.0 * norm, 25.0)
    plotter.camera.focal_point = tuple(focus)
    plotter.camera.position = tuple(focus + direction * distance)
    plotter.camera.up = tuple(np.cross(direction, axis))
    plotter.camera.reset_clipping_range()
