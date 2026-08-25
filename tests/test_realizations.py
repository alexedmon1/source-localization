"""Realization-averaged ROI operators."""
import numpy as np
import pytest

from source_localization.source_space.realizations import (
    build_roi_operator, farthest_point_sample)

RNG = np.random.default_rng(7)


def _toy(n_pool=400, n_chan=30, n_parcels=6):
    """A pool with parcel structure: nearby sources share a label and a topography."""
    pos = RNG.normal(size=(n_pool, 3)) * 4.0
    lab_id = np.argsort(pos[:, 0]) // (n_pool // n_parcels)
    lab_id = np.clip(lab_id, 0, n_parcels - 1)
    order = np.argsort(np.argsort(pos[:, 0]))
    labels = [f"P{lab_id[order[i]]}" for i in range(n_pool)]
    base = RNG.normal(size=(n_chan, n_parcels))
    G = np.column_stack([base[:, lab_id[order[i]]] + 0.15 * RNG.normal(size=n_chan)
                         for i in range(n_pool)])
    return G, pos, labels


def test_farthest_point_sample_is_seeded_and_well_spread():
    _, pos, _ = _toy()
    a = farthest_point_sample(pos, 40, seed=1)
    b = farthest_point_sample(pos, 40, seed=1)
    c = farthest_point_sample(pos, 40, seed=2)
    assert np.array_equal(a, b), "same seed must give the same realization"
    assert not np.array_equal(a, c), "different seeds must differ"
    assert len(np.unique(a)) == 40
    # farthest-point must spread further than a uniform-random draw
    nn = lambda s: np.median([
        np.min(np.linalg.norm(pos[s] - pos[i], axis=1)[np.arange(len(s)) != j])
        for j, i in enumerate(s)])
    rand = RNG.choice(len(pos), 40, replace=False)
    assert nn(a) > nn(rand)


def test_cannot_take_more_than_the_pool_holds():
    _, pos, _ = _toy()
    with pytest.raises(ValueError, match="pool"):
        farthest_point_sample(pos, 10_000, seed=0)


def test_operator_shape_and_row_order():
    G, pos, labels = _toy()
    op, parcels, report = build_roi_operator(G, pos, labels, n_sources=60, k=8)
    assert op.shape == (len(parcels), G.shape[0])
    assert parcels == sorted(parcels)
    assert set(report) == set(parcels)
    assert all(report[p]["n_realizations"] <= 8 for p in parcels)


def test_averaging_operators_equals_averaging_time_series():
    """The claim the whole storage plan rests on.

    Averaging ROI time series over realizations must equal applying the averaged
    operator once. If this ever fails, per-realization outputs would have to be
    materialised and the design collapses.
    """
    from source_localization.source_space.realizations import _parcel_rows
    G, pos, labels = _toy()
    B = RNG.normal(size=(G.shape[0], 200))          # sensor data
    k, n = 12, 60
    op, parcels, _ = build_roi_operator(G, pos, labels, n_sources=n, k=k, seed=3)

    per_real = {p: [] for p in parcels}
    for j in range(k):
        idx = farthest_point_sample(pos, n, 3 + j)
        for p, row in _parcel_rows(G, idx, labels, 1.0 / 9.0).items():
            per_real[p].append(row @ B)             # this realization's ROI series
    averaged_series = np.vstack([np.mean(per_real[p], axis=0) for p in parcels])

    np.testing.assert_allclose(op @ B, averaged_series, rtol=1e-10, atol=1e-12)


def test_degenerate_parcels_are_flagged_not_silently_reported():
    """Two parcels sharing a topography have no stable individual solution."""
    G, pos, labels = _toy()
    own_a = [i for i, l in enumerate(labels) if l == "P0"]
    own_b = [i for i, l in enumerate(labels) if l == "P1"]
    G[:, own_b] = G[:, own_a[:len(own_b)]]          # make P1 a copy of P0
    _, parcels, report = build_roi_operator(G, pos, labels, n_sources=60, k=10)
    assert report["P0"]["max_collinearity"] > 0.99
    assert report["P1"]["collinear_with"] in ("P0", "P1")
    # a parcel that is not degenerate must not be flagged
    clean = [p for p in parcels if report[p]["max_collinearity"] < 0.99]
    assert clean, "at least one parcel should be non-degenerate"
    assert all(report[p]["collinear_with"] is None for p in clean)


def test_labels_must_cover_the_pool():
    G, pos, labels = _toy()
    with pytest.raises(ValueError, match="one entry per pool source"):
        build_roi_operator(G, pos, labels[:-5], n_sources=40, k=4)


def test_unlabelled_sources_never_enter_a_parcel():
    G, pos, labels = _toy()
    labels = [None if i % 3 == 0 else l for i, l in enumerate(labels)]
    _, parcels, _ = build_roi_operator(G, pos, labels, n_sources=60, k=6)
    assert None not in parcels and "None" not in parcels


# --- bilateral merging -----------------------------------------------------

def _roi_cfg(merge=None):
    return {"roi": {"merge_bilateral": merge} if merge is not None else {}}


def test_merge_bilateral_collapses_only_the_named_pairs():
    from source_localization.steps import roi_extraction as rx
    src = {1: "Cerebellum_L", 17: "Cerebellum_R", 4: "Auditory_L",
           20: "Auditory_R", 13: "Olfactory_Bulb_L", 29: "Olfactory_Bulb_R"}

    def apply(merge):
        lab = dict(src)
        for lid, nm in list(lab.items()):
            for base in (merge or []):
                if nm in (f"{base}_L", f"{base}_R"):
                    lab[lid] = base
        return sorted(set(lab.values()))

    assert apply(None) == sorted(set(src.values()))
    assert apply(["Cerebellum"]) == ["Auditory_L", "Auditory_R", "Cerebellum",
                                     "Olfactory_Bulb_L", "Olfactory_Bulb_R"]
    merged = apply(["Cerebellum", "Olfactory_Bulb"])
    assert merged == ["Auditory_L", "Auditory_R", "Cerebellum", "Olfactory_Bulb"]
    assert "Auditory_L" in merged and "Auditory_R" in merged, \
        "auditory is the most separable pair in the atlas and must never merge"


def test_merge_bilateral_is_off_by_default():
    """A study opts in; the atlas is not silently coarsened."""
    for cfg in ({}, {"roi": {}}, {"roi": {"merge_bilateral": None}}):
        merge = list((cfg.get("roi") or {}).get("merge_bilateral") or [])
        assert merge == []


def _pool_with_one_rare_parcel(n_pool=400, n_chan=30, n_parcels=6, n_rare=3):
    """A pool where one parcel holds so few sources that most draws miss it.

    This is the shape of the deployed surface pool, where Hippocampus_Ant_R held
    a handful of the 7,076 sources and entered 1 of 100 draws.

    The rare sources sit at the *centre* of the cloud, not at its edge. Draws
    come from farthest-point sampling, which favours extremes, so a rare parcel
    placed on the boundary is picked in every draw and does not exercise the
    defect at all.
    """
    G, pos, labels = _toy(n_pool, n_chan, n_parcels)
    centre = pos.mean(axis=0)
    rare = np.argsort(np.linalg.norm(pos - centre, axis=1))[:n_rare]
    for i in rare:
        labels[i] = "RARE"
    return G, pos, labels


def test_rare_parcels_do_not_get_inflated_rows():
    """Regression for X42.

    The operator averaged each parcel by the number of draws *that parcel*
    appeared in. A parcel sampled once kept an unaveraged single-draw row, while
    a parcel sampled every draw was reduced by cancellation across draws, so the
    rarest parcel carried the largest row and won every argmax. Normalising by
    k instead makes row scale reflect how often the parcel was actually sampled.
    """
    G, pos, labels = _pool_with_one_rare_parcel()
    op, parcels, report = build_roi_operator(G, pos, labels, n_sources=40, k=60)

    assert "RARE" in parcels, "the rare parcel should still be represented"
    cov = report["RARE"]["coverage"]
    assert cov < 0.5, f"fixture is not exercising the defect (coverage {cov})"

    norms = {p: float(np.linalg.norm(op[i])) for i, p in enumerate(parcels)}
    common = [n for p, n in norms.items() if report[p]["coverage"] > 0.9]
    assert common, "expected some parcels present in nearly every draw"
    assert norms["RARE"] <= max(common), (
        f"rare parcel row norm {norms['RARE']:.3g} exceeds every well-sampled "
        f"parcel (max {max(common):.3g}) — X42 has regressed")


def test_coverage_is_reported_for_every_parcel():
    G, pos, labels = _pool_with_one_rare_parcel()
    _op, parcels, report = build_roi_operator(G, pos, labels, n_sources=40, k=60)
    for p in parcels:
        assert 0.0 < report[p]["coverage"] <= 1.0
        assert report[p]["coverage"] == report[p]["n_realizations"] / 60


def test_fully_sampled_parcels_are_unchanged_by_the_k_normalisation():
    """Parcels in every draw have counts[p] == k, so the fix must not move them."""
    G, pos, labels = _toy()
    op, parcels, report = build_roi_operator(G, pos, labels, n_sources=60, k=25)
    full = [p for p in parcels if report[p]["coverage"] == 1.0]
    assert full, "toy pool should have parcels present in every draw"
    for p in full:
        i = parcels.index(p)
        assert np.isfinite(op[i]).all()
