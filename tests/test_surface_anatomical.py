"""Surface method switch and anatomical surface invariants.

The switch tests are the important ones for regression: the icosphere path is
the baseline every anatomical result is measured against, and MS1 Table 2 was
computed on it, so it must stay bit-identical as the anatomical path evolves.
"""

from __future__ import annotations

import numpy as np
import pytest

from source_localization.source_space import surface


BEM = {
    "center_mm": [0.0, -0.5, -0.3],
    "semi_axes_mm": [5.5, 7.5, 4.5],
    "rotation_matrix": np.eye(3).tolist(),
    "brain_radius_mm": 5.5,
}


def _icosphere_config(**extra):
    cfg = {"ico_level": 3, "inset_factor": 0.80, "filter_dorsal": True}
    cfg.update(extra)
    return {
        "source_space": {"surface": cfg},
        "pipeline": {"bem_type": "ellipsoid"},
    }


def test_default_method_is_icosphere_and_unchanged():
    """No `method` key must behave exactly as before the switch existed."""
    default = surface.create_source_space(_icosphere_config(), {"bem_params": BEM})
    explicit = surface.create_source_space(
        _icosphere_config(method="icosphere"), {"bem_params": BEM}
    )

    src_d, coords_d, n_d = default
    src_e, coords_e, n_e = explicit

    assert n_d == n_e
    assert np.array_equal(coords_d, coords_e)
    assert np.array_equal(src_d[0]["rr"], src_e[0]["rr"])
    assert np.array_equal(src_d[0]["nn"], src_e[0]["nn"])
    assert np.array_equal(src_d[0]["tris"], src_e[0]["tris"])


def test_icosphere_still_yields_the_published_source_count():
    """MS1 Table 2 reports 305 sources for ellipsoid+surface at these settings."""
    _, _, n = surface.create_source_space(_icosphere_config(), {"bem_params": BEM})
    assert n == 305


def test_icosphere_remains_a_single_source_space():
    """Only the anatomical path splits into two hemispheres."""
    src, _, _ = surface.create_source_space(_icosphere_config(), {"bem_params": BEM})
    assert len(src) == 1


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown surface method"):
        surface.create_source_space(
            _icosphere_config(method="not_a_method"), {"bem_params": BEM}
        )


@pytest.mark.slow
def test_anatomical_hemispheres_are_symmetric_by_construction():
    """Mirroring is what removes the atlas's 20-29% per-parcel L/R asymmetry."""
    from source_localization.source_space import surface_anatomical

    lh, rh, meta = surface_anatomical.build_hemispheres(
        spacing_mm=0.8, verbose=False
    )

    assert lh.n_points == rh.n_points
    assert lh.area == pytest.approx(rh.area, abs=1e-9)

    # right is the left reflected through x = 0, vertex for vertex
    lp = np.asarray(lh.points, dtype=float)
    rp = np.asarray(rh.points, dtype=float)
    assert np.allclose(rp[:, 0], -lp[:, 0])
    assert np.allclose(rp[:, 1:], lp[:, 1:])

    # normals reflect with the geometry
    ln = np.asarray(lh.point_data["Normals"], dtype=float)
    rn = np.asarray(rh.point_data["Normals"], dtype=float)
    assert np.allclose(rn[:, 0], -ln[:, 0])
    assert np.allclose(rn[:, 1:], ln[:, 1:])

    nn = np.asarray(lh.point_data["Normals"], dtype=float)
    assert np.allclose(np.linalg.norm(nn, axis=1), 1.0)

    # every cortical parcel survives; this is the shell's failure mode (X15)
    for parcel in (1, 2, 3, 4, 5, 6, 14):
        assert parcel in meta["parcels_present"], f"parcel {parcel} missing"

    # L/R label mismatch is the atlas's own asymmetry, not the mesh's, and it
    # is computed over labelled vertices only. Measured 4.6-6.2% across
    # 0.15-0.80 mm spacing, so it should not drift with resolution.
    assert 0.0 <= meta["label_mismatch_frac"] < 0.15

    # Unlabelled fraction DOES rise as spacing coarsens, because boundary
    # vertices survive decimation preferentially and boundary vertices are the
    # ones sitting on unlabelled voxels. Measured 2.6% at 0.15 mm, 4.3% at
    # 0.30 mm (the working spacing), 16.0% here at a deliberately coarse
    # 0.80 mm with only ~190 vertices per hemisphere. Bound reflects the test's
    # spacing rather than the operating point.
    assert meta["unlabelled_frac"] < 0.20


@pytest.mark.slow
def test_anatomical_source_space_shape_and_units():
    cfg = {
        "source_space": {"surface": {"method": "anatomical", "spacing_mm": 0.8}},
        "pipeline": {"bem_type": "ellipsoid"},
    }
    prev = {}
    src, coords_mm, n = surface.create_source_space(cfg, prev)

    assert len(src) == 2
    assert [s["id"] for s in src] == [101, 102]
    assert all(s["type"] == "surf" for s in src)
    assert n == len(coords_mm) == sum(s["np"] for s in src)

    # rr is metres for MNE while coords stay millimetres for the pipeline
    rr = np.vstack([s["rr"] for s in src])
    assert np.abs(rr).max() < 0.02
    assert np.abs(coords_mm).max() > 1.0

    nn = np.vstack([s["nn"] for s in src])
    assert np.allclose(np.linalg.norm(nn, axis=1), 1.0)

    assert len(prev["surface_parcels"]) == n

    # roi_extraction reads assignments off src[0] and takes their presence as
    # "already assigned", which is the land-in-parcel rule this surface uses
    assert "roi_assignments" in src[0]
    assert len(src[0]["roi_assignments"]) == n
    assert len(src[0]["hemi_roi_assignments"]) == src[0]["np"]
    assert len(src[1]["hemi_roi_assignments"]) == src[1]["np"]


@pytest.mark.slow
def test_electrode_proximity_filter_keeps_both_hemispheres():
    """The filter rebuilt only src[0], which would silently drop the right hemisphere.

    Dormant today because no preset sets max_electrode_distance_mm, but the
    step's docstring advertises the filter as universal.
    """
    from source_localization.steps.source_space import _rebuild_source_space

    cfg = {
        "source_space": {"surface": {"method": "anatomical", "spacing_mm": 0.8}},
        "pipeline": {"bem_type": "ellipsoid"},
    }
    src, _, n = surface.create_source_space(cfg, {})
    lh_n = src[0]["np"]

    keep = np.ones(n, dtype=bool)
    keep[::7] = False
    out = _rebuild_source_space(src, keep)

    assert len(out) == 2
    assert [s["id"] for s in out] == [101, 102]
    assert out[0]["np"] == int(keep[:lh_n].sum())
    assert out[1]["np"] == int(keep[lh_n:].sum())
    assert sum(s["np"] for s in out) == int(keep.sum())

    # whole-space assignments filter with the full mask, per-hemi with theirs
    assert len(out[0]["roi_assignments"]) == int(keep.sum())
    assert len(out[0]["hemi_roi_assignments"]) == out[0]["np"]
    assert len(out[1]["hemi_roi_assignments"]) == out[1]["np"]

    for s in out:
        if s["ntri"]:
            assert s["tris"].max() < s["np"]


def test_electrode_proximity_filter_single_entry_unchanged():
    from source_localization.steps.source_space import _rebuild_source_space

    src, _, n = surface.create_source_space(
        _icosphere_config(), {"bem_params": BEM}
    )
    keep = np.ones(n, dtype=bool)
    keep[::5] = False
    out = _rebuild_source_space(src, keep)

    assert len(out) == 1
    assert out[0]["np"] == int(keep.sum())
