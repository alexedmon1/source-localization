"""The method vocabulary: what this package can actually be asked to do.

One declaration per choice, so the CLI's ``choices=``, its ``list`` output and
the documentation all read from the same place. ``tests/test_methods.py`` checks
each name against the dispatcher that has to accept it, so a method added to
``steps/inverse_solution.py`` and not here (or the reverse) fails the suite
rather than going unlisted -- ``--method`` offered three of the six inverse
methods for several releases because the list was written out by hand.

A *pipeline* is four independent choices: a head model (BEM), where the sources
go (source space), how densely they are placed (sampling), and how the inverse
is solved. The atlas is a fifth, and lives in ``data/atlas/registry.yaml``.
"""

from __future__ import annotations

from typing import Dict, NamedTuple


class Method(NamedTuple):
    """One selectable option, with what a user needs to choose between them."""

    name: str
    summary: str
    #: Longer note: when to prefer it, or what it is known not to do.
    detail: str = ""
    #: True for the value presets use unless they say otherwise.
    default: bool = False
    #: Set when the option carries a caveat that changes how results are read.
    caveat: str = ""


# --- Head models -----------------------------------------------------------

BEM_TYPES: Dict[str, Method] = {
    'sphere': Method(
        'sphere', '3-layer analytical sphere',
        'Closed-form, so it is fast and never fails to converge. The mouse head '
        'is not spherical, and roi_based source spaces lose sources to it.'),
    'ellipsoid': Method(
        'ellipsoid', 'numerical ellipsoid BEM',
        'Fits the mouse head; ~1.5 mm localization error in simulation. '
        'Slower to build, and cached under bem_cache/.', default=True),
}


# --- Where the sources go --------------------------------------------------

SOURCE_SPACES: Dict[str, Method] = {
    'surface': Method(
        'surface', 'cortical mid-ribbon mesh, or an icosphere',
        "source_space.surface.method: 'anatomical' (default) cuts the ribbon "
        "from Allen32 and carries fixed orientation; 'icosphere' is the older "
        "geometric shell.", default=True),
    'roi_based': Method(
        'roi_based', 'one source per atlas ROI, at its PCA centroid',
        'The conventional comparator. Excluded from Monte Carlo sampling: its '
        'placement is the method, so there is no dense pool to draw from.'),
    'cartesian': Method(
        'cartesian', 'regular 3D grid through the brain volume',
        'Grid phase is anchored to voxel 0 of the volume array, a bounding-box '
        'corner with no anatomical meaning.',
        caveat='Sweeping that offset across one cell moves parcel coverage '
               'between 29 and 32. Prefer monte_carlo sampling here.'),
    'shell': Method(
        'shell', 'concentric geometry-matched shells',
        'Depth-stratified sampling that follows the head shape.'),
}


# --- How densely, and at which placement -----------------------------------

SAMPLING_MODES: Dict[str, Method] = {
    'fixed': Method(
        'fixed', 'solve one source grid',
        'Produces vertex-level output as well as parcel series. The grid is one '
        'arbitrary placement of a continuous current distribution.',
        default=True),
    'monte_carlo': Method(
        'monte_carlo', 'average the ROI operator over K sparse draws',
        'Integrates over source placement instead of committing to one grid. '
        'Nearly free: the parcel series is linear in the sensor data, so K '
        'operators are averaged and applied once. Works with surface, '
        'cartesian and shell.',
        caveat='ROI-only by construction — no single grid is solved, so the run '
               'has no vertex output and skip_roi_extraction raises.'),
}


# --- How the inverse is solved ---------------------------------------------

INVERSE_METHODS: Dict[str, Method] = {
    'sLORETA': Method(
        'sLORETA', 'standardized low-resolution tomography',
        'The default in every preset, and the most robust to mouse-scale '
        'leadfields.', default=True),
    'dSPM': Method(
        'dSPM', 'noise-normalized minimum norm'),
    'MNE': Method(
        'MNE', 'raw minimum norm'),
    'eLORETA': Method(
        'eLORETA', 'exact low-resolution tomography',
        'Custom iterative implementation; slower than the three above.'),
    'LCMV': Method(
        'LCMV', 'linearly constrained minimum variance beamformer (stock MNE)',
        caveat='Not covered by the mouse-scale regularization fix that the four '
               'minimum-norm methods above get.'),
    'DICS': Method(
        'DICS', 'dynamic imaging of coherent sources (stock MNE)',
        'Frequency-domain beamformer.',
        caveat='One power value per source, not a time course: written as CSV, '
               'and it fails on the surface presets. Not covered by the '
               'mouse-scale regularization fix.'),
}


def default_of(options: Dict[str, Method]) -> str | None:
    """The name marked default, or None when the choice has no default."""
    return next((m.name for m in options.values() if m.default), None)


def choices(options: Dict[str, Method]) -> list:
    """Names in declaration order, for ``argparse(choices=...)``."""
    return list(options)
