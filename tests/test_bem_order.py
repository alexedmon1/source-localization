"""The surface BEM against MNE's analytic sphere (bem/solution.py).

MNE's solver assumes the outermost surface first; this package builds surfaces inner first. These pin:
- ``mne_order`` sorts head, outer skull, inner skull whatever the input order, and leaves the caller's list alone;
- a three-shell numerical sphere of mouse size (brain 5.75 mm, skull +0.5 mm, scalp +0.3 mm, the preset's
  conductivities) solved through ``make_bem_solution`` matches the analytic sphere at ico-3: topography error
  (RDM, average reference) median under 0.03 and amplitude within 10%;
- the same surfaces passed to MNE inner first do not (the fault this module fixes), so the test can fail;
- surfaces whose ids disagree with their nesting (stale [4, 3, 1] caches, descending sphere radii) are refused.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")
from source_localization.bem.ellipsoid import create_ellipsoid_surfaces_from_axes  # noqa: E402
from source_localization.bem.solution import MNE_ORDER, make_bem_solution, mne_order  # noqa: E402

R, LAYERS, SIG = 5.75, (0.5, 0.3), [0.33, 0.0042, 0.33]


def _setup():
    axes = [np.full(3, R)]
    for t in LAYERS:
        axes.append(axes[-1] * (1 + t / axes[-1].mean()))
    surfs = create_ellipsoid_surfaces_from_axes(np.zeros(3), axes, np.eye(3), SIG, n_vertices=642)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(24, 3))
    d[:, 2] = np.abs(d[:, 2]) + 1
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    names = [f"E{i}" for i in range(24)]
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, d * axes[-1][0] / 1000)), coord_frame="head"))
    pos = rng.normal(size=(20, 3)) * 1.2 / 1000                      # deep, within ~2.5 mm of the centre
    src = mne.setup_volume_source_space(pos=dict(rr=pos, nn=np.tile([0, 0, 1.0], (len(pos), 1))), verbose=False)
    sph = mne.make_sphere_model(r0=(0, 0, 0), head_radius=axes[-1][0] / 1000,
                                relative_radii=[a[0] / axes[-1][0] for a in axes], sigmas=SIG, verbose=False)
    return surfs, info, src, sph


def _gain(info, src, bem):
    g = mne.make_forward_solution(info, trans=None, src=src, bem=bem, eeg=True, meg=False, mindist=0.0,
                                  verbose=False)["sol"]["data"]
    return (g - g.mean(0)).T


def _rdm_mag(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return np.median(np.linalg.norm(a / na[:, None] - b / nb[:, None], axis=1)), np.median(na / nb)


def test_mne_order_sorts_and_copies():
    surfs, *_ = _setup()
    before = [s["id"] for s in surfs]
    for perm in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
        assert [s["id"] for s in mne_order([surfs[i] for i in perm])] == list(MNE_ORDER)
    assert [s["id"] for s in surfs] == before


def test_surface_bem_matches_analytic_sphere():
    surfs, info, src, sph = _setup()
    ref = _gain(info, src, sph)
    rdm, mag = _rdm_mag(_gain(info, src, make_bem_solution(surfs)), ref)
    assert rdm < 0.03 and 0.9 < mag < 1.1, (rdm, mag)
    assert [s["id"] for s in surfs][0] == MNE_ORDER[-1]               # the builder's own order is untouched
    rdm_bad, mag_bad = _rdm_mag(_gain(info, src, mne.make_bem_solution(surfs, verbose=False)), ref)
    assert rdm_bad > 0.5 or not 0.5 < mag_bad < 2, (rdm_bad, mag_bad)


def test_ids_that_disagree_with_nesting_are_refused():
    surfs, *_ = _setup()
    stale = [dict(s) for s in surfs]
    for s, i in zip(stale, MNE_ORDER):                                  # [4, 3, 1] on inner-first geometry
        s["id"] = i
    with pytest.raises(ValueError, match="not nested"):
        make_bem_solution(stale)


def test_descending_sphere_radii_are_refused():
    from source_localization.bem.sphere import create_numerical_sphere_surfaces
    surfs = create_numerical_sphere_surfaces(np.zeros(3), [6.55, 6.25, 5.75], SIG[::-1], n_vertices=642)
    with pytest.raises(ValueError, match="not nested"):
        make_bem_solution(surfs)
