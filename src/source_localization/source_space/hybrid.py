"""Hybrid source space: the anatomical cortical surface plus a volume grid for deep structures.

A surface source space reaches only the structures that have a surface -- cortex,
cerebellum, olfactory bulb. Everything interior (thalamus, tectum, hippocampus,
basal ganglia, hypothalamus, amygdala) has nowhere to go, so a response generated
there is forced onto the nearest cortex. On the MEA30 dorsal array the auditory
evoked responses fit a posterior-midline deep source better than any cortical one
(surface-evoked F023), which makes that the case that matters.

This puts the two together, the way MNE's "mixed" source spaces do. It is called
hybrid rather than mixed because "mixed model" already means something else to
anyone reading the statistics.

  * **Surface part** -- exactly `surface_anatomical`: the Allen mid-ribbon, two
    entries (ids 101/102), sources oriented along the cortical normal.
  * **Volume part** -- the Cartesian grid (`cartesian_based`, same brain mask and
    BEM inset), kept only where the atlas label belongs to a category the surface
    does not carry. One entry (id 1), no normals, so orientation is free.

Cortex is represented once, by the surface: grid points labelled with a surface
category are dropped, not duplicated. A grid point in an unlabelled voxel takes
the nearest labelled voxel's label, the rule `roi_extraction` and the Cartesian
Monte Carlo pool already use; the count is printed.

Monte Carlo sampling only. Under `monte_carlo` every pool source becomes one
leadfield column -- surface sources along their normal, volume sources along
their dominant leadfield direction, the Cartesian arm's existing collapse -- and
that is handled in `steps/monte_carlo_roi.py`. The fixed-grid inverse assumes one
orientation rule for every source and has no path for a mix, so `fixed` sampling
raises here rather than silently treating the surface as free.

Config::

    source_space:
      surface: {...}                  # as for source_type: surface
      hybrid:
        volume_categories: [brainstem, thalamic, hypothalamic, hippocampal, subcortical]
        volume_spacing_mm: 0.4        # grid spacing of the volume pool
      source_sampling: monte_carlo
      monte_carlo:
        pool_spacing_mm: 0.2          # the surface pool, as for source_type: surface

`volume_categories` defaults to every category of the atlas's ``roi_mapping``
that `source_space.surface.categories` leaves out.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mne
import nibabel as nib
import numpy as np

VOLUME_ENTRY_ID = 1


def _surface_categories(config):
    from .surface_anatomical import DEFAULT_CATEGORIES
    surf = config["source_space"].get("surface") or {}
    return tuple(surf.get("categories", DEFAULT_CATEGORIES))


def volume_category_ids(config):
    """(categories, label ids) the volume part carries, from the atlas's roi_mapping."""
    pkg = Path(__file__).resolve().parent.parent
    mapping = json.loads((pkg / config["inputs"]["roi_mapping"]).read_text())
    cats = mapping.get("categories")
    if not cats:
        raise ValueError(
            f"{config['inputs']['roi_mapping']} defines no categories, so the hybrid "
            "source space cannot tell surface structures from deep ones")
    surface = set(_surface_categories(config))
    hyb = config["source_space"].get("hybrid") or {}
    wanted = hyb.get("volume_categories")
    if wanted is None:
        wanted = [c for c in cats if c not in surface]
    unknown = [c for c in wanted if c not in cats]
    if unknown:
        raise ValueError(f"volume_categories {unknown} are not in the atlas's categories {sorted(cats)}")
    overlap = sorted(set(wanted) & surface)
    if overlap:
        raise ValueError(
            f"categories {overlap} are in both the surface and the volume part; "
            "each structure must be represented once")
    return list(wanted), sorted({int(i) for c in wanted for i in cats[c]})


def _volume_labels(config, coords_mm):
    """Atlas label id per grid point: the voxel's own, else the nearest labelled voxel's."""
    from ..steps.roi_extraction import map_sources_to_rois_nearest
    from ..utils.atlas import get_true_affine

    pkg = Path(__file__).resolve().parent.parent
    nii = nib.load(pkg / config["inputs"]["brain_labels"])
    data = nii.get_fdata()
    ids = {int(v) for v in np.unique(data) if v > 0}
    by_id = map_sources_to_rois_nearest(coords_mm, data, get_true_affine(nii), {i: i for i in ids})
    labels = np.zeros(len(coords_mm), dtype=int)
    for i, idx in by_id.items():
        labels[idx] = i
    return labels


def create_source_space(config, previous_outputs):
    """Surface entries (101, 102) followed by one volume entry for deep structures."""
    from . import cartesian_based, surface_anatomical

    from .pool import is_pool_sampling
    if not is_pool_sampling(config):
        raise ValueError(
            "source_type 'hybrid' supports source_sampling: monte_carlo or parcel_subspace "
            "only. The "
            "fixed-grid inverse applies one orientation rule to every source, and a "
            "hybrid needs fixed orientation on the surface and free in the volume.")

    categories, deep_ids = volume_category_ids(config)
    hyb = config["source_space"].get("hybrid") or {}
    spacing = float(hyb.get("volume_spacing_mm", 0.4))

    print("  Creating hybrid source space:")
    print("   [surface part]")
    src_s, coords_s, n_s = surface_anatomical.create_source_space(config, previous_outputs)
    surf_parcels = np.asarray(src_s[0]["roi_assignments"], dtype=int)

    print(f"   [volume part] categories {categories}")
    vcfg = copy.deepcopy(config)
    vcfg["source_space"]["cartesian"] = dict(
        (config["source_space"].get("cartesian") or config["source_space"].get("volumetric") or {}),
        spacing_mm=spacing)
    vcfg["source_space"].pop("volumetric", None)
    for key in ("monte_carlo", "parcel_subspace"):
        if key == "monte_carlo" or config["source_space"].get(key) is not None:
            vcfg["source_space"][key] = dict(
                config["source_space"].get(key) or {}, pool_spacing_mm=spacing)
    src_v, coords_v, _ = cartesian_based.create_source_space(vcfg, previous_outputs)
    vlab = _volume_labels(config, coords_v)
    keep = np.isin(vlab, deep_ids)
    print(f"    Kept {keep.sum():,} of {len(keep):,} grid points in deep structures "
          f"({len(keep) - keep.sum():,} lie in surface structures or outside the atlas)")
    if not keep.any():
        raise ValueError("no grid point falls in the volume categories")
    coords_v, vlab = coords_v[keep], vlab[keep]
    n_v = len(coords_v)

    vol = dict(src_v[0])
    vol.update({
        "rr": coords_v / 1000.0,
        "nn": np.zeros((n_v, 3)),
        "inuse": np.ones(n_v, dtype=int),
        "vertno": np.arange(n_v),
        "nuse": n_v,
        "np": n_v,
        "id": VOLUME_ENTRY_ID,
        "hemi_roi_assignments": vlab,
    })

    lh, rh = src_s[0], src_s[1]
    # roi_assignments on src[0] spans the whole source space (lh, rh, volume),
    # the convention forward_solution and the electrode filter rely on.
    lh["roi_assignments"] = np.concatenate([surf_parcels, vlab])
    src = mne.SourceSpaces([lh, rh, vol])
    coords = np.vstack([coords_s, coords_v])

    previous_outputs["hybrid_parts"] = {"n_surface": int(n_s), "n_volume": int(n_v),
                                        "volume_categories": categories}
    print(f"    ✓ Hybrid: {n_s:,} surface + {n_v:,} volume = {len(coords):,} sources")
    return src, coords, len(coords)


def pool_columns(fwd, orientation="fixed"):
    """One leadfield column per source of a hybrid forward, and which part it is in.

    Surface sources take the normal component when ``orientation`` is ``fixed``;
    otherwise, and always for volume sources, the dominant direction of the
    source's three columns (largest singular vector, scaled by its singular
    value) -- the collapse the Cartesian Monte Carlo pool uses.

    Returns ``G (n_channels, n_sources)`` and a bool array, True for surface.
    """
    G3 = fwd["sol"]["data"]
    n = G3.shape[1] // 3
    if fwd["source_ori"] != mne.io.constants.FIFF.FIFFV_MNE_FREE_ORI or G3.shape[1] != 3 * n:
        raise ValueError("pool_columns needs the free-orientation forward")
    is_surf = np.concatenate([np.full(int(s["nuse"]), s["type"] == "surf") for s in fwd["src"]])
    if len(is_surf) != n:
        raise ValueError(f"forward has {n} sources, its source space {len(is_surf)}")
    blocks = G3.reshape(G3.shape[0], n, 3)
    out = np.empty((G3.shape[0], n))
    if orientation == "fixed":
        # The normals live on the source space. A free-orientation forward's own
        # `source_nn` is the x/y/z axes repeated, not the anatomy.
        nn = np.vstack([np.asarray(s["nn"])[np.asarray(s["vertno"])] for s in fwd["src"]])
        out[:, is_surf] = np.einsum("cnk,nk->cn", blocks[:, is_surf], nn[is_surf])
        rest = ~is_surf
    elif orientation == "free":
        rest = np.ones(n, bool)
    else:
        raise ValueError(f"hybrid orientation must be fixed or free, got {orientation}")
    for i in np.flatnonzero(rest):
        u, s, _ = np.linalg.svd(blocks[:, i, :], full_matrices=False)
        out[:, i] = u[:, 0] * s[0]
    return out, is_surf
