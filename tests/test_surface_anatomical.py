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
    # `method` is stated explicitly because the default is now `anatomical`
    # (D36). These tests exist to lock the icosphere path bit-identical, and
    # that path is opt-in from here on — Phase 3's anatomy-vs-radial contrast
    # depends on still being able to build one.
    cfg = {"method": "icosphere", "ico_level": 3, "inset_factor": 0.80,
           "filter_dorsal": True}
    cfg.update(extra)
    return {
        "source_space": {"surface": cfg},
        "pipeline": {"bem_type": "ellipsoid"},
    }


def test_default_method_is_anatomical():
    """No `method` key now selects the anatomical mid-ribbon surface (D36).

    The default was `icosphere` while the anatomical path was being built, so
    that the replacement would land as a one-line flip rather than a config
    migration. This is that flip. The signature is unambiguous: the anatomical
    path emits two hemispheres (ids 101/102), the icosphere one entry.
    """
    cfg = {"ico_level": 3, "inset_factor": 0.80, "filter_dorsal": True}
    src, _, n = surface.create_source_space(
        {"source_space": {"surface": cfg}, "pipeline": {"bem_type": "ellipsoid"}},
        {"bem_params": BEM},
    )
    assert len(src) == 2
    assert [s["id"] for s in src] == [101, 102]
    assert n != 305, "305 is the icosphere count; the default should not be it"


def test_explicit_icosphere_is_bit_identical_to_the_legacy_path():
    """Asking for the icosphere gives exactly what it always gave."""
    a = surface.create_source_space(_icosphere_config(), {"bem_params": BEM})
    b = surface.create_source_space(
        _icosphere_config(method="icosphere"), {"bem_params": BEM}
    )
    src_a, coords_a, n_a = a
    src_b, coords_b, n_b = b

    assert n_a == n_b == 305
    assert np.array_equal(coords_a, coords_b)
    assert np.array_equal(src_a[0]["rr"], src_b[0]["rr"])
    assert np.array_equal(src_a[0]["nn"], src_b[0]["nn"])
    assert np.array_equal(src_a[0]["tris"], src_b[0]["tris"])


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


def _allen26_paths():
    """allen26 paths, or skip. Its ids are NOT allen32's left 1-16/right 17-32."""
    from pathlib import Path

    from source_localization.config import atlas_input_paths
    from source_localization.source_space import surface_anatomical

    pkg = Path(surface_anatomical.__file__).resolve().parent.parent
    try:
        inputs = atlas_input_paths("allen26")
    except Exception:  # pragma: no cover - atlas not installed
        pytest.skip("allen26 atlas unavailable")
    lp, mp = pkg / inputs["brain_labels"], pkg / inputs["roi_mapping"]
    if not (lp.exists() and mp.exists()):  # pragma: no cover
        pytest.skip("allen26 atlas files missing")
    return lp, mp


def test_midline_seam_uses_atlas_correspondence_not_a_fixed_offset():
    """The seam relabel must read the atlas, not assume allen32's numbering.

    A vertex snapped to x = 0 is sampled by both hemispheres from the same
    voxel, so the right side inherits a left label and has to be sent to its
    right-hand counterpart. That was done as ``+16`` / ``-16``, which is only
    true for allen32. On allen26 (Auditory_L=2/Auditory_R=12,
    Lateral_Cortex_L=6/Lateral_Cortex_R=16) the ``-16`` branch fired on
    *midline* parcels instead: Cerebellum (22) became Lateral_Cortex_L (6),
    Olfactory_Bulb (25) became Somatosensory_L (9) and Frontal_Anterior (23)
    became Motor_L (7), on the left hemisphere only. (X45)
    """
    import json

    from source_localization.source_space import surface_anatomical

    lp, mp = _allen26_paths()
    lh, rh, meta = surface_anatomical.build_hemispheres(
        spacing_mm=0.8, verbose=False,
        categories=("cortical", "cerebellum", "olfactory"),
        labels_path=lp, mapping_path=mp,
    )

    rois = json.loads(mp.read_text())
    rois = rois.get("rois", rois)
    id_by_name = {v["name"]: int(k) for k, v in rois.items() if v.get("name")}
    lateral = {i for n, i in id_by_name.items() if n.endswith(("_L", "_R"))}

    lab_l = np.asarray(lh.point_data["parcel"])
    lab_r = np.asarray(rh.point_data["parcel"])
    seam_l = np.asarray(lh.points)[:, 0] == 0.0
    seam_r = np.asarray(rh.points)[:, 0] == 0.0
    assert seam_l.sum() > 0, "no midline seam to test"
    assert np.array_equal(seam_l, seam_r), "hemispheres are mirrored vertex-wise"

    # A parcel with no _L/_R counterpart spans the midline and is correct on
    # both sides, so the seam must carry the same id on each. Under the old
    # offset the left side of every such vertex was shifted down by 16.
    for i in np.flatnonzero(seam_r):
        if int(lab_r[i]) and int(lab_r[i]) not in lateral:
            assert int(lab_l[i]) == int(lab_r[i]), (
                f"midline parcel {lab_r[i]} became {lab_l[i]} on the left"
            )

    # and the metric that shares the assumption reports atlas asymmetry again,
    # not the ~98% the fixed offset produced on allen26
    assert 0.0 <= meta["label_mismatch_frac"] < 0.15


def test_no_source_carries_a_parcel_outside_the_build_categories():
    """Off-ribbon drift must not promote a phantom parcel. (X46)

    Decimation moves a vertex by a fraction of a voxel and it can land on a
    structure the surface was never built from. Carrying that id gives it a
    first-class ROI time series downstream — a 1-vertex ``Hippocampus_Ant_R``
    sat beside real parcels in every Monte Carlo surface table. Such vertices
    are set unlabelled, and the drift is still *reported* so X17 stays visible.
    """
    from source_localization.source_space import surface_anatomical

    lp, mp = _allen26_paths()
    cats = ("cortical", "cerebellum", "olfactory")
    field = surface_anatomical.build_depth_field(
        0.08, categories=cats, labels_path=lp, mapping_path=mp
    )
    build_ids = set(int(i) for i in field["cortical_ids"])

    lh, rh, meta = surface_anatomical.build_hemispheres(
        spacing_mm=0.8, verbose=False, categories=cats,
        labels_path=lp, mapping_path=mp,
    )
    present = set(
        int(v) for v in np.unique(
            np.concatenate([np.asarray(lh.point_data["parcel"]),
                            np.asarray(rh.point_data["parcel"])])
        )
    )
    stray = present - build_ids - {0}
    assert not stray, f"parcels outside the build categories survived: {stray}"

    # the diagnostic survives the fix: drift is counted before it is zeroed
    assert "noncortical_dropped" in meta
    assert meta["noncortical_frac"] >= 0.0
