"""Orientation-constrained inverse.

The load-bearing test here is that ``free`` is bit-identical to the code path
that produced every published number. Everything else in this project is
measured against that baseline, so if it drifts, the comparison is void.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from source_localization.source_space import surface
from source_localization.steps import inverse_solution as inv

mne.set_log_level("ERROR")

SPACING = 0.8  # coarse on purpose: correctness, not characterisation


@pytest.fixture(scope="module")
def fwd_and_n():
    cfg = {
        "source_space": {"surface": {"method": "anatomical", "spacing_mm": SPACING}},
        "pipeline": {"bem_type": "ellipsoid"},
    }
    src, _, n_sources = surface.create_source_space(cfg, {})

    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0), head_radius=0.0075,
        relative_radii=(0.87, 0.92, 1.0), sigmas=(0.33, 0.0042, 0.33),
        verbose=False,
    )
    rng = np.random.default_rng(0)
    phi = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    theta = np.arccos(rng.uniform(0.15, 0.95, size=30))
    pos = 0.0075 * np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    names = [f"E{i+1}" for i in range(30)]
    info = mne.create_info(names, sfreq=500.0, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, pos)), coord_frame="head"
    ))
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                    eeg=True, meg=False, n_jobs=1, verbose=False)
    return fwd, n_sources


@pytest.mark.slow
def test_free_is_bit_identical_to_the_legacy_path(fwd_and_n):
    """This is the baseline every anatomical result is compared against."""
    fwd, _ = fwd_and_n
    G = fwd["sol"]["data"]
    n_ch = G.shape[0]

    lambda2 = 1.0 / 3.0 ** 2
    GGT = G @ G.T
    GGT_reg = GGT + lambda2 * np.trace(GGT) / n_ch * np.eye(n_ch)
    W_legacy = G.T @ np.linalg.inv(GGT_reg)
    norm_legacy = np.sqrt(np.sum(W_legacy * G.T, axis=1))

    W, normalizer, n_comp = inv.compute_inverse_operator(
        fwd, method="sLORETA", snr=3.0, orientation="free", verbose=False
    )

    assert n_comp == 3
    assert np.array_equal(W, W_legacy)
    assert np.array_equal(normalizer, norm_legacy)


@pytest.mark.slow
def test_fixed_collapses_to_one_component(fwd_and_n):
    fwd, n_sources = fwd_and_n
    W, normalizer, n_comp = inv.compute_inverse_operator(
        fwd, method="sLORETA", snr=3.0, orientation="fixed", verbose=False
    )
    assert n_comp == 1
    assert W.shape[0] == n_sources

    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((W.shape[1], 64))
    magnitude, signed = inv.apply_inverse_to_epoch(
        W, normalizer, epoch, n_sources, n_comp=1
    )
    assert magnitude.shape == signed.shape == (n_sources, 64)
    # the single component is the projection on the normal, so |signed| is the
    # magnitude and its sign is anatomically meaningful
    assert np.allclose(np.abs(signed), magnitude)


@pytest.mark.slow
def test_loose_attenuates_tangential_components(fwd_and_n):
    fwd, n_sources = fwd_and_n
    loose = 0.2
    W_free, _, _ = inv.compute_inverse_operator(
        fwd, method="sLORETA", snr=3.0, orientation="free", verbose=False
    )
    W_loose, _, n_comp = inv.compute_inverse_operator(
        fwd, method="sLORETA", snr=3.0, orientation="loose", loose=loose,
        verbose=False,
    )
    assert n_comp == 3
    assert not np.allclose(W_loose, W_free)

    n_ch = W_free.shape[1]
    ratio_free = _tangential_ratio(W_free, n_sources, n_ch)
    ratio_loose = _tangential_ratio(W_loose, n_sources, n_ch)
    assert ratio_loose < ratio_free
    # the penalty is the loose factor itself, within discretisation
    assert ratio_loose / ratio_free == pytest.approx(loose, rel=0.35)


def _tangential_ratio(W, n_sources, n_ch):
    Wr = W.reshape(n_sources, 3, n_ch)
    normal = np.linalg.norm(Wr[:, 0, :], axis=1).mean()
    tangential = np.linalg.norm(Wr[:, 1:, :], axis=(1, 2)).mean()
    return tangential / normal


@pytest.mark.slow
def test_relative_regularisation_is_orientation_invariant(fwd_and_n):
    """The auto-scaling already absorbs the trace change (D20).

    The regularisation added is lambda2 * trace(GG')/n_ch, so its ratio to the
    signal scale is lambda2 whatever the orientation. Fixed changes the
    absolute term but not the relative one, which is why no regularisation
    sweep is owed here.
    """
    fwd, _ = fwd_and_n
    lambda2 = 1.0 / 9.0
    ratios = []
    for ori in ("free", "loose", "fixed"):
        f2, n_comp, weights = inv.apply_orientation_constraint(
            fwd, orientation=ori, loose=0.2, verbose=False
        )
        G = f2["sol"]["data"]
        if weights is not None:
            G = G * weights[np.newaxis, :]
        n_ch = G.shape[0]
        GGT = G @ G.T
        ratios.append(lambda2 * np.trace(GGT) / n_ch / (np.trace(GGT) / n_ch))
    assert np.allclose(ratios, lambda2)


@pytest.mark.slow
def test_orientation_requires_normals(fwd_and_n):
    fwd, _ = fwd_and_n
    stripped = fwd.copy()
    for s in stripped["src"]:
        s["nn"] = np.zeros_like(s["nn"])

    for ori in ("fixed", "loose"):
        with pytest.raises(ValueError, match="needs source normals"):
            inv.apply_orientation_constraint(stripped, orientation=ori, verbose=False)

    # free never touches the normals, so it still works
    _, n_comp, weights = inv.apply_orientation_constraint(
        stripped, orientation="free", verbose=False
    )
    assert n_comp == 3 and weights is None


def test_unknown_orientation_raises():
    with pytest.raises(ValueError, match="Unknown orientation"):
        inv.apply_orientation_constraint(None, orientation="sideways")


def test_loose_out_of_range_raises(fwd_and_n):
    fwd, _ = fwd_and_n
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="loose must be in"):
            inv.apply_orientation_constraint(
                fwd, orientation="loose", loose=bad, verbose=False
            )
