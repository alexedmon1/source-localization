"""The surface BEM against MNE's analytic sphere (bem/solution.py).

MNE's solver assumes the outermost surface first; this package builds surfaces inner first. These pin:
- ``mne_order`` sorts head, outer skull, inner skull whatever the input order, and leaves the caller's list alone;
- a three-shell numerical sphere of mouse size (brain 5.75 mm, skull +0.5 mm, scalp +0.3 mm, the preset's
  conductivities) solved through ``make_bem_solution`` matches the exact multi-shell series at ico-3: topography
  error (RDM, average reference) median under 0.03 and amplitude within 10%. The reference is the series, not
  ``mne.make_sphere_model``: MNE's Berg-Scherg fit fails on this geometry for some last-bit changes in the radii (5-7%)
  (``bem.sphere.check_sphere_fit``), which is how one CI job got NaN leadfields;
- the same surfaces passed to MNE inner first do not (the fault this module fixes), so the test can fail;
- surfaces whose ids disagree with their nesting (stale [4, 3, 1] caches, descending sphere radii) are refused;
- an analytic sphere whose fit failed is refused.
"""
import numpy as np
import pytest
from numpy.polynomial.legendre import legval

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
    return surfs, info, src, (np.array([a[0] for a in axes]), d, pos)


def _series(radii, sig, beta, n_max):
    """Per Legendre order n, the potential at the outer surface of concentric shells (insulating outside) for a
    unit point source at relative radius beta in the innermost shell, without the 1 / (4 pi sig_1 R) factor.
    Unknowns A_1, (A_k, B_k): V = beta^n / rho^(n+1) + A_1 rho^n in shell 1, A_k rho^n + B_k rho^-(n+1) beyond;
    V and sig dV/drho continuous at each interface, dV/drho = 0 at rho = 1."""
    rho, N = np.asarray(radii, float) / radii[-1], len(radii)
    col = lambda k, b: 0 if k == 0 else 2 * k - 1 + b
    out = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        M, rhs = np.zeros((2 * N - 1, 2 * N - 1)), np.zeros(2 * N - 1)
        for j in range(N - 1):
            r = rho[j]
            for eq in range(2):
                a = lambda s: r ** n if eq == 0 else s * n * r ** (n - 1)
                b = lambda s: r ** -(n + 1) if eq == 0 else -s * (n + 1) * r ** -(n + 2)
                row = 2 * j + eq
                M[row, col(j, 0)] += a(sig[j])
                if j == 0:
                    rhs[row] -= b(sig[j]) * beta ** n
                else:
                    M[row, col(j, 1)] += b(sig[j])
                M[row, col(j + 1, 0)] -= a(sig[j + 1])
                M[row, col(j + 1, 1)] -= b(sig[j + 1])
        if N == 1:                                       # A_1 only: n A_1 = (n + 1) beta^n
            M[-1, 0], rhs[-1] = n, (n + 1) * beta ** n
        else:
            M[-1, col(N - 1, 0)], M[-1, col(N - 1, 1)] = n, -(n + 1)
        scale = np.abs(M).max(0)
        x = np.linalg.solve(M / scale, rhs) / scale
        out[n] = x[0] + beta ** n if N == 1 else x[col(N - 1, 0)] + x[col(N - 1, 1)]
    return out


def _exact_gain(radii_mm, el_dirs, pos_m, n_max=250):
    """Exact dipole leadfield (3 rows per source, x y z), average referenced: central differences of the monopole
    potential in the source position (h = 1e-6 R)."""
    R = radii_mm[-1] / 1000.0
    def mono(p):
        b = np.linalg.norm(p)
        return legval(el_dirs @ (p / b), _series(radii_mm, SIG, b / R, n_max) / (4 * np.pi * SIG[0] * R))
    h, rows = 1e-6 * R, []
    for p in pos_m:
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            rows.append((mono(p + e) - mono(p - e)) / (2 * h))
    G = np.array(rows)
    return G - G.mean(1, keepdims=True)


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


def test_exact_series_matches_a_single_sphere():
    # One insulated homogeneous shell has a closed form: for a radial dipole at relative radius beta, the outer
    # potential is sum_n (2n+1) beta^(n-1) P_n(c) / (4 pi sig R^2), which the Legendre generating function sums to
    # [(1 - beta^2) / (1 - 2 beta c + beta^2)^1.5 - 1] / (beta 4 pi sig R^2). Independent of the series solver and
    # of the finite difference, both of which this checks.
    R, sig, beta = 0.01, 0.33, 0.4
    c = np.cos(np.linspace(0.05, 3.1, 25))
    mono = lambda b: legval(c, _series([R * 1000], [sig], b / R, 400) / (4 * np.pi * sig * R))
    h = 1e-6 * R
    g = (mono(beta * R + h) - mono(beta * R - h)) / (2 * h)
    closed = ((1 - beta ** 2) / (1 - 2 * beta * c + beta ** 2) ** 1.5 - 1) / (beta * 4 * np.pi * sig * R ** 2)
    assert np.allclose(g, closed, rtol=1e-6, atol=1e-9 * np.abs(closed).max())


def test_surface_bem_matches_exact_sphere():
    surfs, info, src, (radii, el_dirs, pos) = _setup()
    ref = _exact_gain(radii, el_dirs, pos)
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


def test_failed_sphere_fits_are_refused():
    from source_localization.bem.sphere import check_sphere_fit
    good = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.0066, relative_radii=[0.87, 0.92, 1.0], sigmas=SIG,
                                 verbose=False)
    assert check_sphere_fit(good) is good
    failed = dict(good, mu=np.array([0.5, 0.0, 0.0]), **{"lambda": np.array([2.2, -2.97e16, 2.97e16])})  # as seen on CI
    with pytest.raises(ValueError, match="Berg-Scherg fit"):
        check_sphere_fit(failed)
