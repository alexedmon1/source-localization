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

DEFAULT_ISO_MM = 0.08
DEFAULT_SPACING_MM = 0.30
DEFAULT_MIN_THICKNESS_MM = 0.16
DEFAULT_CORTEX_DILATION_VOX = 1
TAUBIN_ITER = 30
TAUBIN_PASS_BAND = 0.1

_ATLAS_SUBDIR = Path("data") / "atlas"


def _atlas_dir() -> Path:
    return Path(__file__).resolve().parent.parent / _ATLAS_SUBDIR


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "source-localization" / "surfaces"


def build_depth_field(iso_mm=DEFAULT_ISO_MM):
    """Cortical mask, resampled to isotropic, plus the ribbon depth field."""
    import nibabel as nib
    from scipy import ndimage

    adir = _atlas_dir()
    allen = adir / "allen"

    mapping = json.loads((allen / "roi_mapping.json").read_text())
    cortical_ids = list(mapping["categories"]["cortical"])

    labels_img = nib.load(allen / "allen_labels.nii.gz")
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
                      field=None, verbose=True):
    """Return (lh, rh, meta). rh is a mirror of lh, so geometry is symmetric."""
    import pyvista as pv

    if field is None:
        field = build_depth_field(iso_mm=iso_mm)

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
    lh = left.decimate(reduction, volume_preservation=True).clean()
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
    lh.point_data["parcel"] = lab_l.astype(np.int16)
    rh.point_data["parcel"] = lab_r.astype(np.int16)

    mismatch = float((lab_r != lab_l + 16).mean())

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


def _cache_path(cache_dir, spacing_mm, iso_mm):
    tag = f"midribbon_s{spacing_mm:.3f}_i{iso_mm:.3f}".replace(".", "p")
    return Path(cache_dir) / f"{tag}.npz"


def load_or_build(spacing_mm, iso_mm, cache_dir, verbose=True):
    """Build the template surface once and reuse it across subjects.

    The BEM is fitted to the atlas brain mask rather than to each animal, so
    this geometry is identical for every subject in a study. Rebuilding it per
    subject would repeat several minutes of marching cubes for no difference.
    """
    import pyvista as pv

    path = _cache_path(cache_dir, spacing_mm, iso_mm)
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

    lh, rh, meta = build_hemispheres(spacing_mm, iso_mm, verbose=verbose)
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

    print("  Creating anatomical surface source space:")
    print(f"    Target spacing: {spacing_mm} mm")

    (lh_pts, lh_nn, lh_tris, lh_parcel,
     rh_pts, rh_nn, rh_tris, rh_parcel, meta) = load_or_build(
        spacing_mm, iso_mm, cache_dir
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
