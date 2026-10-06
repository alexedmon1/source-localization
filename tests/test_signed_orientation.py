"""The signed series under free orientation: per-epoch axis choice against a fixed direction (X56).

By default `apply_inverse_to_epoch` takes, for each source and each epoch, whichever of its three
components has the largest variance in that epoch, so a source can switch axis between epochs.
On real 40 Hz ASSR recordings that happens in about 15% of epochs and costs phase locking
(surface-evoked troubleshooting log, section 28). `inverse.signed_orientation: leadfield` projects
every epoch onto a fixed direction per source instead. The default must stay bit-identical.
"""
import numpy as np

from source_localization.steps import inverse_solution as inv


def _legacy_signed(W, normalizer, epoch, n_sources):
    a = W @ epoch
    if normalizer is not None:
        a = a / (normalizer[:, None] + 1e-10)
    a = a.reshape(n_sources, 3, -1)
    idx = np.var(a, axis=2).argmax(axis=1)
    return a[np.arange(n_sources), idx].astype(np.float32)


def test_default_is_unchanged():
    rng = np.random.default_rng(0)
    n_src, n_ch = 5, 8
    W = rng.normal(size=(3 * n_src, n_ch))
    norm = np.abs(rng.normal(size=3 * n_src)) + 0.5
    epoch = rng.normal(size=(n_ch, 40))
    _, signed = inv.apply_inverse_to_epoch(W, norm, epoch, n_src, n_comp=3)
    np.testing.assert_array_equal(signed, _legacy_signed(W, norm, epoch, n_src))


def test_fixed_dirs_project_every_epoch():
    rng = np.random.default_rng(1)
    n_src, n_ch = 4, 6
    W = rng.normal(size=(3 * n_src, n_ch))
    epoch = rng.normal(size=(n_ch, 30))
    dirs = rng.normal(size=(n_src, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mag, signed = inv.apply_inverse_to_epoch(W, None, epoch, n_src, n_comp=3, fixed_dirs=dirs)
    a = (W @ epoch).reshape(n_src, 3, -1)
    np.testing.assert_allclose(signed, np.einsum("sct,sc->st", a, dirs), rtol=1e-5)
    np.testing.assert_allclose(mag, np.linalg.norm(a, axis=1), rtol=1e-5)


def test_fixed_dirs_keep_phase_locking_where_axis_switching_loses_it():
    """One source, a 10 Hz oscillation along a fixed oblique direction whose x and y components have
    opposite signs. Noise makes the largest-variance axis alternate between x and y across epochs,
    and each switch flips the series' phase by 180 degrees; a fixed direction does not."""
    rng = np.random.default_rng(2)
    t = np.arange(200) / 200.0
    d = np.array([0.7, -0.7, 0.14])
    d /= np.linalg.norm(d)
    W = np.eye(3)                                               # identity operator: data = activity
    coef = {"per_epoch": [], "fixed": []}
    basis = np.exp(-2j * np.pi * 10 * t)
    for _ in range(60):
        act = d[:, None] * np.cos(2 * np.pi * 10 * t)[None] + rng.normal(scale=0.6, size=(3, t.size))
        _, s0 = inv.apply_inverse_to_epoch(W, None, act, 1, n_comp=3)
        _, s1 = inv.apply_inverse_to_epoch(W, None, act, 1, n_comp=3, fixed_dirs=d[None])
        coef["per_epoch"].append(s0[0] @ basis)
        coef["fixed"].append(s1[0] @ basis)
    itc = {k: abs(np.mean(np.array(v) / np.abs(v))) for k, v in coef.items()}
    assert itc["fixed"] > itc["per_epoch"] + 0.05


def _fwd(G):
    return {"sol": {"data": G}}


def test_leadfield_directions_are_unit_dominant_and_dorsal():
    rng = np.random.default_rng(3)
    n_ch, n_src = 10, 6
    G = rng.normal(size=(n_ch, 3 * n_src))
    dirs = inv.leadfield_directions(_fwd(G))
    assert dirs.shape == (n_src, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0)
    for s in range(n_src):
        blk = G[:, 3 * s:3 * s + 3]
        v = np.linalg.svd(blk)[2][0]
        assert abs(abs(v @ dirs[s]) - 1.0) < 1e-10           # the dominant direction, up to sign
    assert np.all(dirs[:, 2] >= 0)                            # dorsal
    # deterministic in the sign: flipping the leadfield does not flip the direction
    np.testing.assert_allclose(inv.leadfield_directions(_fwd(-G)), dirs)


def test_parcel_alignment_makes_sources_agree_within_a_parcel():
    """Two sources with opposite leadfields form one parcel: unaligned, their patterns cancel in a
    mean; aligned, they agree. Sources without a parcel keep the dorsal anchor."""
    rng = np.random.default_rng(4)
    n_ch = 8
    base = rng.normal(size=(n_ch, 3))
    G = np.hstack([base, -base * 1.1, rng.normal(size=(n_ch, 3))])     # sources 0, 1 opposite; 2 alone
    labels = ["P", "P", None]
    plain = inv.leadfield_directions(_fwd(G))
    aligned = inv.leadfield_directions(_fwd(G), labels)
    topo = lambda d: np.einsum("csk,sk->cs", G.reshape(n_ch, 3, 3), d)
    assert topo(aligned)[:, 0] @ topo(aligned)[:, 1] > 0
    np.testing.assert_allclose(np.abs(aligned), np.abs(plain))           # same directions, signs only
    np.testing.assert_allclose(aligned[2], plain[2])                      # unlabelled: dorsal anchor kept
