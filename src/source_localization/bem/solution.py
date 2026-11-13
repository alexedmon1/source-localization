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


def make_bem_solution(surfs, verbose=False):
    """``mne.make_bem_solution`` on the surfaces in MNE's order. The caller's list and dicts are not modified."""
    return mne.make_bem_solution(copy.deepcopy(mne_order(surfs)), verbose=verbose)
