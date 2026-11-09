"""Defining an electrode field from group data.

A *field* is a fixed pattern over the electrodes (unit norm, average-referenced) that carries a response. The field
method defines it once from a group's data, freezes it as a *field template* (:mod:`.template`), and then projects
every recording onto it (``field project``) and locates it (``field locate``). Each definition here is catalogued
(``methods.FIELD_DEFINITIONS``):

- ``field.response_svd`` (:func:`response_svd`): the leading spatial singular vectors of the group-mean response over
  a time span, each unit's response first scaled to unit norm over that span so every unit weighs the same. Used for
  the chirp (surface-evoked troubleshooting log 67).
- ``field.phase_aligned`` (:func:`phase_aligned_pattern`): for a steady-state response's complex coefficients, each
  recording is rotated onto the current pattern's phase and scaled to unit norm, recordings are averaged per unit,
  and the real part of the units' mean becomes the new pattern; repeated. Used for the 40 and 80 Hz ASSR (log 69).
- ``field.phase_free`` (:func:`phase_free_pattern`): the leading eigenvector of sum_r Re(z_r z_r^H) / |z_r|^2, the
  same quantity without an iteration or a starting pattern (surface-evoked ``method_crossover.FieldModel``).
- ``field.csd_leading`` (:func:`csd_leading_pattern`): the leading eigenvector of the mean trace-normalised
  cross-spectral density, for an ongoing (resting) band (``method_crossover.FieldModel``, bayes-sl L2/L3).

Every function returns unit-norm maps in the channel order it was given. Signs are arbitrary until a sign rule is
applied (:func:`apply_sign_rule`); the rule is part of the template, never a constant in code.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["scale_units", "response_svd", "phase_aligned_pattern", "phase_free_pattern", "csd_leading_pattern",
           "apply_sign_rule", "unit_index"]


def unit_index(keys: Sequence) -> Tuple[list, list]:
    """Group recordings into units by key (for example (mouse, day)): sorted unique keys and, per key, its indices."""
    keys = [tuple(k) if isinstance(k, (list, tuple)) else (k,) for k in keys]
    uniq = sorted(set(keys))
    return uniq, [[i for i, k in enumerate(keys) if k == u] for u in uniq]


def scale_units(responses: np.ndarray, span: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scale each unit's response (U, C, T) by its Frobenius norm over the time mask ``span``. Returns scaled, scales."""
    responses = np.asarray(responses, float)
    scales = np.linalg.norm(responses[:, :, span].reshape(len(responses), -1), axis=1)
    return responses / scales[:, None, None], scales


def response_svd(group_mean: np.ndarray, span: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """``field.response_svd``: the first ``k`` spatial singular vectors (C, k) of ``group_mean[:, span]`` and all
    singular values. Pass the mean of :func:`scale_units`' output as ``group_mean``."""
    U, s, _ = np.linalg.svd(np.asarray(group_mean, float)[:, span], full_matrices=False)
    return U[:, :k], s


def phase_aligned_pattern(y: np.ndarray, units: Sequence[Sequence[int]], init: np.ndarray, n_iter: int = 3,
                          noise: Optional[np.ndarray] = None):
    """``field.phase_aligned``: real pattern of complex steady-state coefficients ``y`` (R, C).

    Each iteration: normalise the pattern D; rotate each recording by conj(y.D)/|y.D| and divide by ||y|| (onto D's
    phase, unit norm); average the rotated recordings within each unit (``units``: index lists); D = real part of the
    units' mean. After ``n_iter`` iterations D is normalised and the recordings are aligned once more with it.

    Returns D (C,), the aligned unit means (n_units, C) and, if ``noise`` (R, k, C) is given, the aligned unit noise
    (n_units, k, C). The result depends on ``init`` only through which phase it aligns to.
    """
    y = np.asarray(y, complex)
    D = np.asarray(init, float)

    def align(D):
        z = y @ D
        rot = np.conj(z) / np.abs(z) / np.linalg.norm(y, axis=1)
        Y = y * rot[:, None]
        U = np.array([Y[j].mean(0) for j in units])
        N = None if noise is None else np.array([(noise * rot[:, None, None])[j].mean(0) for j in units])
        return U, N

    for _ in range(n_iter):
        D = D / np.linalg.norm(D)
        U, _ = align(D)
        D = U.mean(0).real
    D = D / np.linalg.norm(D)
    U, N = align(D)
    return D, U, N


def phase_free_pattern(y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``field.phase_free``: leading eigenvector of sum_r w_r Re(z_r z_r^H) / |z_r|^2 over complex patterns (R, C)."""
    y = np.asarray(y, complex)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    M = sum(wi * np.real(np.outer(z, z.conj())) / np.vdot(z, z).real for wi, z in zip(w, y))
    return np.linalg.eigh(M)[1][:, -1]


def csd_leading_pattern(csd: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``field.csd_leading``: leading eigenvector of the weighted mean of trace-normalised CSDs (S, C, C)."""
    csd = np.asarray(csd)
    w = np.ones(len(csd)) if weights is None else np.asarray(weights, float)
    M = sum(wi * np.real(c) / np.real(np.trace(c)) for wi, c in zip(w, csd)) / w.sum()
    return np.linalg.eigh(M)[1][:, -1]


def apply_sign_rule(field: np.ndarray, channels: Sequence[str], rule: Mapping) -> np.ndarray:
    """Flip ``field`` so that the sum over ``rule["electrodes"]`` has the sign ``rule["sign"]`` (+1 or -1)."""
    idx = [list(channels).index(c) for c in rule["electrodes"]]
    s = float(np.sum(np.asarray(field)[idx]))
    if s == 0:
        raise ValueError(f"sign rule electrodes {rule['electrodes']} sum to 0 on this field")
    return np.asarray(field) * (1.0 if np.sign(s) == np.sign(rule.get("sign", 1)) else -1.0)
