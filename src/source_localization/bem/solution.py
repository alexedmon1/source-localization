"""Solve a surface BEM with the surfaces in the order MNE's solver assumes.

``mne.make_bem_solution`` takes the surfaces in list order and assumes the first is the outermost: it puts zero
conductivity outside ``surfs[0]`` and treats the last as the inner skull. MNE's own constructor
(``mne.bem._surfaces_to_bem``) sorts them head, outer skull, inner skull; the solver does not, and does not look at
the ids. This package builds its surfaces inner first (brain, skull, scalp), and every consumer that reads
``bem['surfs'][0]`` as the brain relies on that, so the builders keep their order and the solve goes through here.

Until 2026-11-13 the surfaces were passed to the solver inner first, which solves a physically inconsistent model
(insulator just outside the brain surface). ``SOLVER_TAG`` is part of the cached-operator signature so operators and
ensembles computed that way are not served again.
"""
from __future__ import annotations

import copy

import mne
from mne.io.constants import FIFF

MNE_ORDER = (FIFF.FIFFV_BEM_SURF_ID_HEAD, FIFF.FIFFV_BEM_SURF_ID_SKULL, FIFF.FIFFV_BEM_SURF_ID_BRAIN)
SOLVER_TAG = "surfaces outer-first (bem.solution)"


def mne_order(surfs):
    """The surfaces sorted outermost first by id (a shallow copy of the list; one surface is returned as is)."""
    surfs = list(surfs)
    if len(surfs) == 1:
        return surfs
    ids = [int(s["id"]) for s in surfs]
    if sorted(ids) != sorted(int(i) for i in MNE_ORDER):
        raise ValueError(f"expected one head, one outer-skull and one inner-skull surface, got ids {ids}")
    return [surfs[ids.index(int(i))] for i in MNE_ORDER]


def check_nesting(ordered):
    """Raise unless each surface lies wholly inside the one before it (MNE's own topology check).

    ``mne_order`` trusts the ids. A model whose ids disagree with its geometry (caches written before 7c4ba64, which
    carry ids [4, 3, 1] on inner-first surfaces, or a sphere built from descending radii) would sort into the wrong
    order without this. ``mne.bem._check_surfaces`` is what ``mne.bem._surfaces_to_bem`` runs; that constructor is
    not used here because its size guard rejects mouse-sized heads.
    """
    if len(ordered) != 3:
        return
    try:
        with mne.utils.use_log_level("WARNING"):
            mne.bem._check_surfaces(copy.deepcopy(ordered))
    except RuntimeError as e:
        ids = [int(s["id"]) for s in ordered]
        raise ValueError(f"BEM surfaces are not nested in the order their ids give ({ids}, outermost first): {e}. "
                         "A stale cache or mislabelled surfaces; rebuild the BEM.") from e


def make_bem_solution(surfs, verbose=False):
    """``mne.make_bem_solution`` on the surfaces in MNE's order, after checking they nest in that order.

    The caller's list and dicts are not modified.
    """
    ordered = copy.deepcopy(mne_order(surfs))
    check_nesting(ordered)
    return mne.make_bem_solution(ordered, verbose=verbose)


# ------------------------------------------------------------------ saved forward models
#
# A forward model saved before the order fix looks exactly like one saved after it, and several readers (and batch
# runs that skip finished subjects) reuse saved ones. So every forward the pipeline makes carries the solver that
# made it, in the object and in a sidecar next to step4_forward.pkl, and readers refuse a forward without one. Analytic
# spheres were never affected by the order, but their old forwards carry no tag either and are refused the same way:
# recomputing one takes seconds, and a reader cannot tell them apart.

ANALYTIC_TAG = "analytic sphere (mne.make_sphere_model)"
CURRENT_TAGS = (SOLVER_TAG, ANALYTIC_TAG)
TAG_KEY = "source_localization_bem_solver"
TAG_FILE = "step4_forward_solver.json"
FIX_NOTE = ("Forward models made before source-localization b5de19f (2026-11-13) solved the surface BEM with its "
            "surfaces inner first and are wrong by orders of magnitude; rerun the pipeline, into a new output "
            "directory if the old results are to be kept.")


class StaleForwardError(ValueError):
    """A saved forward model (or something computed from one) with no record of a current BEM solver."""


def tag_forward(fwd, tag):
    """Record on ``fwd`` which solver made it; returns ``fwd``."""
    if tag not in CURRENT_TAGS:
        raise ValueError(f"unknown solver tag {tag!r}")
    fwd[TAG_KEY] = tag
    return fwd


def check_forward(fwd, where="forward model"):
    """Raise :class:`StaleForwardError` unless ``fwd`` carries a current solver tag; returns ``fwd``."""
    tag = fwd.get(TAG_KEY) if hasattr(fwd, "get") else None
    if tag not in CURRENT_TAGS:
        raise StaleForwardError(f"{where} has no current BEM-solver tag ({tag!r}). {FIX_NOTE}")
    return fwd


def load_forward(path):
    """Read a pickled forward model (``step4_forward.pkl``) and refuse it unless it carries a current solver tag."""
    import pickle
    from pathlib import Path
    with open(Path(path), "rb") as f:
        fwd = pickle.load(f)
    return check_forward(fwd, where=str(path))


def write_forward_tag(data_dir, tag, bem_type):
    """Write the sidecar that says which solver made this run's forward (written even without intermediates)."""
    import json
    from pathlib import Path
    from .. import __version__
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / TAG_FILE).write_text(json.dumps({"bem_solver": tag, "bem_type": bem_type,
                                          "source_localization_version": __version__}, indent=1))
    return d / TAG_FILE


def run_forward_is_current(data_dir):
    """True if a run's data directory records a current solver (sidecar, else the pickled forward's own tag)."""
    import json
    from pathlib import Path
    d = Path(data_dir)
    if (d / TAG_FILE).exists():
        try:
            return json.loads((d / TAG_FILE).read_text()).get("bem_solver") in CURRENT_TAGS
        except (OSError, ValueError):
            return False
    if (d / "step4_forward.pkl").exists():
        try:
            load_forward(d / "step4_forward.pkl")
            return True
        except StaleForwardError:
            return False
    return False
