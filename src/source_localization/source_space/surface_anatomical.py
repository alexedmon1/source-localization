"""Anatomical cortical surface source space.

Builds sources on the mid-thickness of the cortical ribbon derived from the
Allen32 parcellation, the way a FreeSurfer surface is derived from an MRI,
rather than on an inset icosphere. The distinction that matters is orientation:
a vertex here carries the local cortical normal, so an orientation-constrained
inverse has something anatomical to constrain to.

Construction
------------
1. Cortical mask from the Allen32 ``cortical`` category (14 of 32 parcels).
2. Resample to isotropic voxels; the native grid is 0.203 x 0.080 x 0.200 mm,
   a 2.5x anisotropy that would otherwise bias triangle quality by axis.
3. Normalised depth ``t = d_out / (d_out + d_in)`` where ``d_out`` is distance
   to the exterior and ``d_in`` distance to interior non-cortex. The three
   masks partition the volume, so t runs 0 at the pial-equivalent boundary to 1
   at the white-matter-equivalent boundary and interpolates only across the
   ribbon. It is computed globally: restricting it to the cortex mask would
   make t jump at every parcel boundary and shatter the isosurface there.
4. Marching cubes at t = 0.5. Where cortex is absent the exterior abuts
   interior non-cortex directly and the level set still appears as a sliver, so
   faces survive only if all three vertices sit in cortex and the local ribbon
   clears a thickness floor.
5. Taubin smoothing, cut at the midline, decimate the left hemisphere, mirror
   it to make the right. Mirroring makes the geometry symmetric by
   construction; the alternative inherits a 20-29% per-parcel L/R asymmetry
   from the atlas.
6. Normals oriented down the depth gradient. Winding order is unreliable on a
   sheet cut open at the midline, and auto-orientation assumes a closed
   manifold, whereas ``-grad(t)`` points to the pial side everywhere.

Parcel labels are sampled from the atlas and are **not** mirrored, so they
carry the atlas's own L/R asymmetry. ``label_mismatch_frac`` in the returned
metadata reports it and must travel with any left-versus-right result.

**Created:** 2026-08-13
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from ..utils.atlas import get_true_affine, get_true_voxel_sizes

# Which Allen32 categories carry sources. "cortical" alone reproduces the
# original behaviour. Cerebellum is a defensible addition: it is 6.4% of the
# outer shell, it is real EEG signal that otherwise gets misattributed to
# cortex, and Purkinje dendrites run perpendicular to the folial surface, so an
# orientation constraint means the same thing there as in neocortex.
#
# NOT included by default: "brainstem" (Allen32's Brainstem_Tectum, i.e.
# superior and inferior colliculus) — the IC is a primary 40 Hz ASSR generator
# so it is worth having for auditory work, but it has no laminar dipole
# geometry, so a fixed-orientation constraint asserts something untrue there.
# And nothing unlabelled: half the outer shell carries no Allen32 label at all,
# being ventral brainstem and white matter the 32-parcel scheme does not name.
# Those sit a median 4.65 mm from the nearest electrode and cannot be
# attributed to any ROI.
DEFAULT_CATEGORIES = ("cortical",)

DEFAULT_ISO_MM = 0.08
# D31: 0.50 mm, chosen on evidence rather than on accuracy. The inverse's own
# contribution to localisation error is flat at 0.52-0.67 mm across a 20x range
# of source counts, because 30 sensors cap the effective rank regardless; denser
# grids only shrink the discretisation floor, which is not resolution. What
# density does bind is parcel occupancy, and 0.50 mm is the coarsest spacing
# keeping every carried parcel at >= 10 sources.
#
# This was 0.30 while the preset carried 0.50 explicitly. Harmless then, because
# `icosphere` was the default and nothing reached here without a preset; wrong
# now that `anatomical` is the default, since a bare surface request would have
# silently got the density D31 rejected (2,546 sources rather than 1,310).
DEFAULT_SPACING_MM = 0.50
DEFAULT_MIN_THICKNESS_MM = 0.16
DEFAULT_CORTEX_DILATION_VOX = 1
TAUBIN_ITER = 30
TAUBIN_PASS_BAND = 0.1

_ATLAS_SUBDIR = Path("data") / "atlas"


def _atlas_dir() -> Path:
    return Path(__file__).resolve().parent.parent / _ATLAS_SUBDIR


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "source-localization" / "surfaces"


def close_cortical_holes(cortex, labels, cortical_ids, iterations=3):
    """Absorb unlabelled voxels enclosed by cortex into their nearest parcel.

    The Allen32 cortical parcellation is not a solid sheet. 8.1% of the volume
    enclosed by cortex carries no cortical label — 87% of it unlabelled
    background, the rest amygdala and hippocampus bleeding in. The mid-ribbon
    level set genuinely does not exist in those voxels, so they perforate the
    surface, and the face filter then erodes a triangle's width around every
    perforation.

    This closes the mask and gives each newly included voxel the label of its
    nearest cortical voxel.

    Note the resemblance to the proximity assignment retired in 66abe69, and
    the difference: that spread one *source* across every parcel within a
    radius, making parcel membership many-to-one and radius-dependent. This
    repairs the *mask* before any source exists, and each voxel still ends up
    in exactly one parcel. It is bounded to voxels morphologically enclosed by
    cortex, so it cannot grow the sheet outward.

    Returns
    -------
    cortex_out, labels_out, stats
    """
    from scipy import ndimage

    closed = ndimage.binary_closing(cortex, iterations=iterations)
    closed = ndimage.binary_fill_holes(closed)
    added = closed & ~cortex
    if not added.any():
        return cortex, labels, {"added_voxels": 0}

    # nearest cortical voxel for every added voxel, then inherit its label
    _, idx = ndimage.distance_transform_edt(
        ~cortex, return_distances=True, return_indices=True
    )
    labels_out = labels.copy()
    ax, ay, az = idx[0][added], idx[1][added], idx[2][added]
    labels_out[added] = labels[ax, ay, az]

    # anything that still failed to inherit a cortical label is left out
    valid = np.isin(labels_out, list(cortical_ids))
    cortex_out = cortex | (added & valid)
    labels_out[added & ~valid] = labels[added & ~valid]

    stats = {
        "added_voxels": int((added & valid).sum()),
        "rejected_voxels": int((added & ~valid).sum()),
        "cortex_before": int(cortex.sum()),
        "cortex_after": int(cortex_out.sum()),
    }
    return cortex_out, labels_out, stats


def build_depth_field(iso_mm=DEFAULT_ISO_MM, close_holes=0,
                      categories=DEFAULT_CATEGORIES,
                      labels_path=None, mapping_path=None):
    """Source-tissue mask, resampled to isotropic, plus the ribbon depth field.

    Parameters
    ----------
    close_holes : int, default 0
        Iterations of hole closing on the mask before the depth field is
        computed. 0 keeps the parcellation exactly as shipped.
    categories : sequence of str
        Allen32 categories that carry sources. See DEFAULT_CATEGORIES.
    """
    import nibabel as nib
    from scipy import ndimage

    adir = _atlas_dir()
    allen = adir / "allen"

    # The atlas is a parameter, not a constant. This module used to load
    # allen/roi_mapping.json and allen/allen_labels.nii.gz unconditionally, so
    # the anatomical surface was built from Allen32 whatever `--atlas` said, and
    # the parcel ids it cached meant whatever Allen32 meant by them (X41).
    mapping_path = Path(mapping_path) if mapping_path else allen / "roi_mapping.json"
    labels_path = Path(labels_path) if labels_path else allen / "allen_labels.nii.gz"
    mapping = json.loads(mapping_path.read_text())
    categories = tuple(categories) if categories else DEFAULT_CATEGORIES
    unknown = [c for c in categories if c not in mapping["categories"]]
    if unknown:
        raise ValueError(
            f"unknown categories {unknown} for {mapping_path.name}; available: "
            f"{sorted(mapping['categories'])}"
        )
    cortical_ids = [i for c in categories for i in mapping["categories"][c]]

    labels_img = nib.load(labels_path)
    labels = np.asarray(labels_img.dataobj).astype(np.int16)
    vox_mm = np.asarray(get_true_voxel_sizes(labels_img), dtype=float)
    affine = get_true_affine(labels_img)

    brain_img = nib.load(adir / "Atlas_3DRois_brain_srcmask.nii.gz")
    brain = np.asarray(brain_img.dataobj) > 0
    if brain.shape != labels.shape:
        raise ValueError(
            f"brain mask {brain.shape} does not match labels {labels.shape}"
        )

    zoom = vox_mm / iso_mm
    labels_iso = ndimage.zoom(labels, zoom, order=0)
    brain_iso = ndimage.zoom(brain.astype(np.uint8), zoom, order=0) > 0
    cortex_iso = np.isin(labels_iso, cortical_ids)

    hole_stats = {"added_voxels": 0}
    if close_holes:
        cortex_iso, labels_iso, hole_stats = close_cortical_holes(
            cortex_iso, labels_iso, cortical_ids, iterations=close_holes
        )

    exterior = ~brain_iso
    interior_noncortex = brain_iso & ~cortex_iso

    sampling = (iso_mm,) * 3
    d_out = ndimage.distance_transform_edt(~exterior, sampling=sampling)
    d_in = ndimage.distance_transform_edt(~interior_noncortex, sampling=sampling)

    denom = d_out + d_in
    depth = np.zeros_like(d_out, dtype=np.float32)
    nz = denom > 0
    depth[nz] = (d_out[nz] / denom[nz]).astype(np.float32)

    iso_affine = affine.copy()
    iso_affine[:3, :3] = np.diag([iso_mm] * 3)

    return {
        "depth": depth,
        "thickness": denom.astype(np.float32),
        "cortex": cortex_iso,
        "labels": labels_iso.astype(np.int16),
        "affine": iso_affine,
        "iso_mm": iso_mm,
        "cortical_ids": cortical_ids,
        "hole_stats": hole_stats,
        "close_holes": close_holes,
        "categories": list(categories),
    }


def extract_midribbon(field, min_thickness_mm=DEFAULT_MIN_THICKNESS_MM,
                      cortex_dilation_vox=DEFAULT_CORTEX_DILATION_VOX):
    """Marching cubes at t = 0.5, with non-cortical slivers rejected."""
    import pyvista as pv
    from scipy import ndimage
    from skimage import measure

    depth = field["depth"]
    cortex = field["cortex"]
    thickness = field["thickness"]
    affine = field["affine"]

    verts_vox, faces, _, _ = measure.marching_cubes(
        depth.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0)
    )

    cortex_ref = (
        ndimage.binary_dilation(cortex, iterations=cortex_dilation_vox)
        if cortex_dilation_vox else cortex
    )

    idx = np.rint(verts_vox).astype(int)
    for ax in range(3):
        idx[:, ax] = np.clip(idx[:, ax], 0, depth.shape[ax] - 1)

    keep = (
        cortex_ref[idx[:, 0], idx[:, 1], idx[:, 2]]
        & (thickness[idx[:, 0], idx[:, 1], idx[:, 2]] >= min_thickness_mm)
    )
    faces = faces[keep[faces].all(axis=1)]

    used = np.unique(faces)
    remap = np.full(len(verts_vox), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts_vox = verts_vox[used]
    faces = remap[faces]

    verts_mm = verts_vox @ affine[:3, :3].T + affine[:3, 3]
    pv_faces = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(verts_mm, pv_faces).clean()


MIDLINE_SNAP_TOL_MM = 0.12


def _snap_midline(mesh, tol_mm=MIDLINE_SNAP_TOL_MM):
    """Pull boundary vertices near x=0 onto x=0 exactly.

    The hemispheres are cut at x=0 and the right is the left mirrored, so the
    two only close if the cut edge sits exactly on the plane. Decimation moves
    it: measured, the left hemisphere reached only x=-0.035 mm, leaving a
    0.07 mm seam down the middle of the merged surface.

    Constrained to vertices already on an open boundary, so interior geometry
    is untouched — snapping every vertex within the tolerance would flatten
    real medial-wall curvature onto the plane.
    """
    boundary = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False,
    )
    if boundary.n_points == 0:
        return mesh

    pts = np.asarray(mesh.points, dtype=float)
    tree = None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, idx = tree.query(np.asarray(boundary.points, dtype=float))
    except Exception:
        return mesh

    on_boundary = np.zeros(len(pts), dtype=bool)
    on_boundary[idx] = True
    snap = on_boundary & (np.abs(pts[:, 0]) <= tol_mm)
    if snap.any():
        pts[snap, 0] = 0.0
        mesh = mesh.copy()
        mesh.points = pts
    return mesh


def _orient_normals(mesh, field):
    """Point every normal to the pial side, using -grad(t)."""
    depth, affine, iso_mm = field["depth"], field["affine"], field["iso_mm"]

    mesh = mesh.compute_normals(
        cell_normals=False, point_normals=True,
        auto_orient_normals=False, consistent_normals=True,
    )
    n = np.asarray(mesh.point_data["Normals"], dtype=float)

    # affine is diagonal, so gradient components come back in world x, y, z
    g0, g1, g2 = np.gradient(depth.astype(np.float32), iso_mm)
    grad = np.stack([g0, g1, g2], axis=-1)

    inv = np.linalg.inv(affine)
    idx = np.rint(mesh.points @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    for ax in range(3):
        idx[:, ax] = np.clip(idx[:, ax], 0, depth.shape[ax] - 1)

    outward = -grad[idx[:, 0], idx[:, 1], idx[:, 2]]
    norm = np.linalg.norm(outward, axis=1, keepdims=True)
    ok = norm[:, 0] > 1e-8
    outward[ok] /= norm[ok]

    n[np.einsum("ij,ij->i", n, outward) < 0] *= -1.0
    mesh.point_data["Normals"] = n
    agree = float(np.abs(np.einsum("ij,ij->i", n[ok], outward[ok])).mean())
    return mesh, agree


def build_hemispheres(spacing_mm=DEFAULT_SPACING_MM, iso_mm=DEFAULT_ISO_MM,
                      field=None, verbose=True, close_holes=0,
                      categories=DEFAULT_CATEGORIES,
                      labels_path=None, mapping_path=None):
    """Return (lh, rh, meta). rh is a mirror of lh, so geometry is symmetric."""
    import pyvista as pv

    if field is None:
        field = build_depth_field(iso_mm=iso_mm, close_holes=close_holes,
                                  categories=categories,
                                  labels_path=labels_path,
                                  mapping_path=mapping_path)

    raw = extract_midribbon(field)
    smoothed = raw.smooth_taubin(
        n_iter=TAUBIN_ITER, pass_band=TAUBIN_PASS_BAND,
        boundary_smoothing=False, normalize_coordinates=True,
    )

    # Clipping emits the cut boundary as line cells and splits triangles into
    # polygons; rebuild from points+faces so decimation gets pure triangles.
    left = smoothed.clip(normal="x", origin=(0.0, 0.0, 0.0), invert=True)
    left = left.extract_surface(algorithm="dataset_surface").triangulate()
    left = pv.PolyData(left.points, left.faces).clean()
    if not left.is_all_triangles:
        raise RuntimeError("left hemisphere is not all triangles after cleanup")

    per_vertex_area = (np.sqrt(3) / 2) * spacing_mm ** 2
    target_n = max(int(left.area / per_vertex_area), 24)
    reduction = float(np.clip(1.0 - target_n / left.n_points, 0.0, 0.999))
    # boundary_constraints keeps the midline cut from being eaten: without it
    # decimation pulls the cut edge inward and the mirrored hemispheres no
    # longer meet.
    lh = left.decimate(reduction, volume_preservation=True,
                       boundary_constraints=True).clean()
    lh = _snap_midline(lh)
    lh, agree = _orient_normals(lh, field)

    rh = lh.copy()
    pts = rh.points.copy()
    pts[:, 0] *= -1.0
    rh.points = pts
    # mirroring reverses handedness; rewind so the surface stays coherent
    f = np.array(rh.faces).reshape(-1, 4)
    f[:, 1:] = f[:, [3, 2, 1]]
    rh.faces = f.ravel()
    nn = np.asarray(lh.point_data["Normals"], dtype=float).copy()
    nn[:, 0] *= -1.0
    rh.point_data["Normals"] = nn

    labels, affine = field["labels"], field["affine"]
    inv = np.linalg.inv(affine)

    def sample(points):
        idx = np.rint(points @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        for ax in range(3):
            idx[:, ax] = np.clip(idx[:, ax], 0, labels.shape[ax] - 1)
        return labels[idx[:, 0], idx[:, 1], idx[:, 2]]

    lab_l, lab_r = sample(lh.points), sample(rh.points)

    # Vertices snapped onto x=0 mirror to themselves, so the right hemisphere
    # samples the very same voxel as the left and inherits a left-hemisphere
    # label. A vertex in the right hemisphere's mesh cannot belong to a left
    # structure by construction, so the seam is relabelled by side. Allen32
    # numbering is left 1-16, right 17-32.
    seam_l = np.asarray(lh.points)[:, 0] == 0.0
    seam_r = np.asarray(rh.points)[:, 0] == 0.0
    flip_r = seam_r & (lab_r >= 1) & (lab_r <= 16)
    lab_r[flip_r] = lab_r[flip_r] + 16
    flip_l = seam_l & (lab_l >= 17) & (lab_l <= 32)
    lab_l[flip_l] = lab_l[flip_l] - 16

    lh.point_data["parcel"] = lab_l.astype(np.int16)
    rh.point_data["parcel"] = lab_r.astype(np.int16)

    # Only meaningful where both sides carry a label. An unlabelled vertex has
    # lab 0 on both sides, and 0 != 0 + 16 counted as a disagreement, so the
    # metric was reporting "the atlas disagrees L/R" for vertices the atlas
    # simply does not label — which the midline seam is full of.
    both_labelled = (lab_l > 0) & (lab_r > 0)
    if both_labelled.any():
        mismatch = float(
            (lab_r[both_labelled] != lab_l[both_labelled] + 16).mean()
        )
    else:
        mismatch = float("nan")

    # Decimation moves vertices off the ribbon, so a few land on unlabelled
    # voxels or, rarely, on a non-cortical parcel. The raw isosurface has none
    # of either. Both are reported rather than snapped: under the land-in-parcel
    # rule a source belongs where it sits, and snapping would reintroduce the
    # proximity assignment that was retired in 66abe69. For scale, gridded
    # source spaces put ~25% of sources on unlabelled voxels.
    cortical = set(field["cortical_ids"])
    both = np.concatenate([lab_l, lab_r])
    unlabelled = float((both == 0).mean())
    noncortical = float(
        (~np.isin(both, list(cortical)) & (both != 0)).mean()
    )

    meta = {
        "spacing_mm": spacing_mm,
        "iso_mm": iso_mm,
        "close_holes": field.get("close_holes", 0),
        "categories": field.get("categories", list(DEFAULT_CATEGORIES)),
        "hole_stats": field.get("hole_stats", {}),
        "n_per_hemi": int(lh.n_points),
        "n_total": int(lh.n_points + rh.n_points),
        "area_per_hemi_mm2": float(lh.area),
        "normal_agreement": agree,
        # D30: mesh is mirrored, labels are not. This is the atlas asymmetry.
        "label_mismatch_frac": mismatch,
        "unlabelled_frac": unlabelled,
        "noncortical_frac": noncortical,
        "parcels_present": sorted(set(int(v) for v in np.unique(lab_l))),
    }

    if verbose:
        print(f"    Anatomical surface: {meta['n_total']:,} sources "
              f"({meta['n_per_hemi']:,} per hemisphere)")
        print(f"      target spacing   : {spacing_mm:.3f} mm")
        print(f"      area per hemi    : {meta['area_per_hemi_mm2']:.2f} mm^2")
        print(f"      normal agreement : {agree:.4f}")
        print(f"      L/R label mismatch: {mismatch * 100:.2f}% "
              "(atlas asymmetry, not mesh)")
        print(f"      unlabelled       : {unlabelled * 100:.2f}%  "
              f"non-cortical: {noncortical * 100:.2f}%")

    return lh, rh, meta


def _cache_path(cache_dir, spacing_mm, iso_mm, close_holes=0,
                categories=DEFAULT_CATEGORIES, atlas_tag=None):
    """Cache filename. `atlas_tag` is part of the key, and must be.

    The cached arrays include `lh_parcel`/`rh_parcel`, which are *atlas label
    ids*. The same id means different structures in different atlases -- 14 is
    Lateral_Cortex_L in Allen32 and Hippocampus_Ant_R in Allen26, and Allen32's
    30 and 31 do not exist in Allen26 at all. Keying the cache on geometry alone
    therefore silently mislabels every source when the atlas changes, and drops
    the sources whose ids the new atlas does not define (X41). Nothing raised;
    the run simply reported the wrong parcels.
    """
    tag = f"midribbon_s{spacing_mm:.3f}_i{iso_mm:.3f}".replace(".", "p")
    if close_holes:
        tag += f"_c{int(close_holes)}"
    cats = tuple(categories) if categories else DEFAULT_CATEGORIES
    if tuple(cats) != DEFAULT_CATEGORIES:
        tag += "_" + "-".join(sorted(c[:4] for c in cats))
    if atlas_tag:
        tag += f"_{atlas_tag}"
    return Path(cache_dir) / f"{tag}.npz"


def load_or_build(spacing_mm, iso_mm, cache_dir, verbose=True,
                  close_holes=0, categories=DEFAULT_CATEGORIES,
                  atlas_tag=None, labels_path=None, mapping_path=None):
    """Build the template surface once and reuse it across subjects.

    The BEM is fitted to the atlas brain mask rather than to each animal, so
    this geometry is identical for every subject in a study. Rebuilding it per
    subject would repeat several minutes of marching cubes for no difference.
    """
    import pyvista as pv

    path = _cache_path(cache_dir, spacing_mm, iso_mm, close_holes,
                       categories, atlas_tag)
    if path.exists():
        d = np.load(path, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        if verbose:
            print(f"    Anatomical surface: cached, {meta['n_total']:,} sources "
                  f"({path})")
        return (
            d["lh_points"], d["lh_normals"], d["lh_faces"], d["lh_parcel"],
            d["rh_points"], d["rh_normals"], d["rh_faces"], d["rh_parcel"],
            meta,
        )

    lh, rh, meta = build_hemispheres(spacing_mm, iso_mm, verbose=verbose,
                                     close_holes=close_holes,
                                     categories=categories,
                                     labels_path=labels_path,
                                     mapping_path=mapping_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def tri(mesh):
        return np.array(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int32)

    payload = dict(
        lh_points=np.asarray(lh.points, dtype=np.float64),
        lh_normals=np.asarray(lh.point_data["Normals"], dtype=np.float64),
        lh_faces=tri(lh),
        lh_parcel=np.asarray(lh.point_data["parcel"], dtype=np.int16),
        rh_points=np.asarray(rh.points, dtype=np.float64),
        rh_normals=np.asarray(rh.point_data["Normals"], dtype=np.float64),
        rh_faces=tri(rh),
        rh_parcel=np.asarray(rh.point_data["parcel"], dtype=np.int16),
        meta=json.dumps(meta),
    )
    np.savez_compressed(path, **payload)
    if verbose:
        print(f"      cached to {path}")
    return (
        payload["lh_points"], payload["lh_normals"], payload["lh_faces"],
        payload["lh_parcel"], payload["rh_points"], payload["rh_normals"],
        payload["rh_faces"], payload["rh_parcel"], meta,
    )


def _hemi_dict(points_mm, normals, faces, src_id):
    n = len(points_mm)
    return {
        "rr": np.ascontiguousarray(points_mm / 1000.0, dtype=np.float64),
        "nn": np.ascontiguousarray(normals, dtype=np.float64),
        "nuse": n,
        "inuse": np.ones(n, dtype=np.int32),
        "vertno": np.arange(n, dtype=np.int32),
        "type": "surf",
        "coord_frame": mne.io.constants.FIFF.FIFFV_COORD_MRI,
        "id": src_id,
        "np": n,
        "ntri": len(faces),
        "nuse_tri": len(faces),
        "tris": np.ascontiguousarray(faces, dtype=np.int32),
        "use_tris": np.ascontiguousarray(faces, dtype=np.int32),
        "nearest": None,
        "nearest_dist": None,
        "pinfo": None,
        "patch_inds": None,
        "dist": None,
        "dist_limit": None,
    }


def create_source_space(config, previous_outputs):
    """Create the anatomical mid-ribbon source space.

    Returns two source spaces, ids 101 (left) and 102 (right), following the
    MNE convention for surface source spaces.
    """
    surface_cfg = config["source_space"].get("surface", {})
    spacing_mm = float(surface_cfg.get("spacing_mm", DEFAULT_SPACING_MM))
    iso_mm = float(surface_cfg.get("iso_mm", DEFAULT_ISO_MM))
    cache_dir = surface_cfg.get("cache_dir") or _default_cache_dir()
    close_holes = int(surface_cfg.get("close_holes", 0))
    categories = tuple(surface_cfg.get("categories", DEFAULT_CATEGORIES))

    print("  Creating anatomical surface source space:")
    print(f"    Target spacing: {spacing_mm} mm")

    # The cached parcel arrays are atlas label ids, so the atlas belongs in the
    # cache key (X41). Identify it by the labels file actually configured rather
    # than by a name, since --atlas rewrites the paths and a preset may point
    # anywhere.
    pkg = Path(__file__).resolve().parent.parent
    inputs = config.get("inputs") or {}
    labels_path = pkg / inputs["brain_labels"] if inputs.get("brain_labels") else None
    mapping_path = pkg / inputs["roi_mapping"] if inputs.get("roi_mapping") else None
    atlas_tag = labels_path.name.split(".")[0] if labels_path else None

    (lh_pts, lh_nn, lh_tris, lh_parcel,
     rh_pts, rh_nn, rh_tris, rh_parcel, meta) = load_or_build(
        spacing_mm, iso_mm, cache_dir, close_holes=close_holes,
        categories=categories, atlas_tag=atlas_tag,
        labels_path=labels_path, mapping_path=mapping_path
    )

    lh_dict = _hemi_dict(lh_pts, lh_nn, lh_tris, 101)
    rh_dict = _hemi_dict(rh_pts, rh_nn, rh_tris, 102)

    parcels = np.concatenate([lh_parcel, rh_parcel])

    # roi_extraction reads `roi_assignments` off src[0] and treats its presence
    # as "sources are already assigned", which also disables proximity
    # assignment — exactly the land-in-parcel rule this surface is built on. The
    # array spans both hemispheres because stc data is ordered lh then rh.
    lh_dict["roi_assignments"] = parcels
    # Per-hemisphere copies so an entry stays self-describing if one is used
    # alone; the electrode-proximity filter slices these per entry.
    lh_dict["hemi_roi_assignments"] = lh_parcel
    rh_dict["hemi_roi_assignments"] = rh_parcel

    src = mne.SourceSpaces([lh_dict, rh_dict])

    coords_mm = np.vstack([lh_pts, rh_pts])
    n_sources = len(coords_mm)

    # Parcel assignment travels with the source space so downstream ROI
    # extraction does not have to re-derive it from coordinates.
    previous_outputs.setdefault("surface_meta", {}).update(meta)
    previous_outputs["surface_parcels"] = parcels

    print(f"    ✓ Created anatomical surface with {n_sources:,} sources "
          f"(lh {len(lh_pts):,} / rh {len(rh_pts):,})")

    return src, coords_mm, n_sources
