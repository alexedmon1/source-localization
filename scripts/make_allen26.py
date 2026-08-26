"""Build Allen26: Allen32 with six inseparable bilateral pairs merged.

A 30-channel dorsal array cannot separate the two halves of a midline structure.
Measured as the cosine between the dominant sensor topographies of each pair,
from a dense sampling of every parcel through the deployed ellipsoid BEM:

    Olfactory_Bulb    0.999      Brainstem_Tectum  0.999
    Cerebellum        0.999      Frontal_Anterior  0.996
    Hypothalamus      0.998      Thalamus          0.992
    ---------------------------------------------------
    Somatosensory     0.184      Auditory          0.099

When a pair is that collinear the inverse splits its shared signal on arbitrary
grounds, and the split is unstable -- it swings with nothing more than where the
source grid happens to land. Reporting such a pair as two parcels invents a
lateralisation the data cannot support. Merging loses nothing recoverable.

Allen26 keeps the ten pairs that ARE separable lateralized, and merges the six
that are not. It is a *montage-specific* parcellation, not a better atlas: a
denser or more lateral array would separate pairs this one cannot. Allen32
remains the default and is unchanged.

    ids  1-10  left  of the ten lateralized structures
    ids 11-20  right of the same, in the same order
    ids 21-26  the six bilateral structures

Writes allen26_labels.nii.gz, roi_mapping_allen26.json, roi_categories_allen26.yaml
and allen26_labels.txt (an ITK-SNAP-style colour LUT) into data/atlas/allen/.

    uv run python scripts/make_allen26.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

MERGE = ["Brainstem_Tectum", "Cerebellum", "Frontal_Anterior",
         "Hypothalamus", "Olfactory_Bulb", "Thalamus"]
COLLINEARITY = {"Olfactory_Bulb": 0.999, "Cerebellum": 0.999,
                "Brainstem_Tectum": 0.999, "Hypothalamus": 0.998,
                "Frontal_Anterior": 0.996, "Thalamus": 0.992}

ATLAS = Path(__file__).resolve().parent.parent / "src/source_localization/data/atlas"
ALLEN = ATLAS / "allen"


def main() -> None:
    src = json.loads((ALLEN / "roi_mapping.json").read_text())
    rois = {int(k): v for k, v in src["rois"].items() if int(k) != 0}
    bases, by_base = [], {}
    for rid, info in sorted(rois.items()):
        base = info["name"].rsplit("_", 1)[0]
        by_base.setdefault(base, {})[info["name"].rsplit("_", 1)[1]] = (rid, info)
        if base not in bases:
            bases.append(base)

    lateral = [b for b in sorted(bases) if b not in MERGE]
    merged = [b for b in sorted(bases) if b in MERGE]
    missing = [b for b in MERGE if b not in bases]
    if missing:
        raise SystemExit(f"not in Allen32: {missing}")
    print(f"  {len(lateral)} lateralized + {len(merged)} bilateral "
          f"-> {2*len(lateral) + len(merged)} labels")

    remap: dict[int, int] = {}
    new_rois: dict[str, dict] = {}

    def carry(info, new_id, name, hemi, extra=None):
        out = {k: v for k, v in info.items()
               if k not in ("id", "name", "hemisphere", "abbreviation")}
        out.update({"id": new_id, "name": name, "hemisphere": hemi,
                    "abbreviation": f"A{new_id:02d}"})
        if extra:
            out.update(extra)
        return out

    for i, base in enumerate(lateral):
        for off, hemi in ((0, "L"), (len(lateral), "R")):
            old_id, info = by_base[base][hemi]
            new_id = 1 + i + off
            remap[old_id] = new_id
            new_rois[str(new_id)] = carry(info, new_id, f"{base}_{hemi}", hemi)

    for j, base in enumerate(merged):
        new_id = 2 * len(lateral) + 1 + j
        (lid, linfo), (rid, rinfo) = by_base[base]["L"], by_base[base]["R"]
        remap[lid] = remap[rid] = new_id
        ids = sorted(set(linfo.get("allen_structure_ids", []))
                     | set(rinfo.get("allen_structure_ids", [])))
        pids = sorted(set(linfo.get("allen_parent_ids", []))
                      | set(rinfo.get("allen_parent_ids", [])))
        new_rois[str(new_id)] = carry(
            linfo, new_id, base, "B",
            {"allen_structure_ids": ids, "allen_parent_ids": pids,
             "merged_from": [linfo["name"], rinfo["name"]],
             "merge_reason": (
                 f"dominant sensor topographies of L and R correlate at "
                 f"{COLLINEARITY[base]:.3f} on the MEA30 dorsal array; the two "
                 f"halves are not separable by this montage")})

    # --- label volume -------------------------------------------------------
    img = nib.load(ALLEN / "allen_labels.nii.gz")
    data = np.asarray(img.dataobj).astype(np.int16)
    out = np.zeros_like(data)
    for old, new in remap.items():
        out[data == old] = new
    if (data > 0).sum() != (out > 0).sum():
        raise SystemExit("labelled voxel count changed; remap is not a bijection "
                         "on the labelled set")
    nib.save(nib.Nifti1Image(out, img.affine, img.header),
             ALLEN / "allen26_labels.nii.gz")

    # --- mapping ------------------------------------------------------------
    mapping = {
        "atlas_name": "Allen Mouse Brain Atlas CCFv3 — Allen26",
        "n_rois": len(new_rois),
        "derived_from": "Allen32",
        "method": ("Allen32 with six bilateral pairs merged, each inseparable by "
                   "the MEA30 dorsal montage (|cos| between dominant sensor "
                   "topographies >= 0.99). Montage-specific, not a better "
                   "parcellation: a denser or more lateral array would separate "
                   "pairs this one cannot."),
        "merged_pairs": {b: COLLINEARITY[b] for b in merged},
        "lateralized": lateral,
        "label_scheme": (f"1-{len(lateral)} left, "
                         f"{len(lateral)+1}-{2*len(lateral)} right (same order), "
                         f"{2*len(lateral)+1}-{len(new_rois)} bilateral"),
        "tiers": src.get("tiers", {}),
        "citation": src.get("citation", ""),
        "rois": {"0": {"id": 0, "name": "Background", "abbreviation": "BG",
                       "color_hex": "000000", "color_rgb": [0, 0, 0]},
                 **new_rois},
    }
    cats: dict[str, list[int]] = {}
    for rid, info in new_rois.items():
        c = info.get("category")
        if c:
            cats.setdefault(c, []).append(int(rid))
    mapping["categories"] = {k: sorted(v) for k, v in sorted(cats.items())}
    (ALLEN / "roi_mapping_allen26.json").write_text(json.dumps(mapping, indent=2))

    # --- categories yaml (names, for source-analytics) ----------------------
    src_cat = yaml.safe_load((ALLEN / "roi_categories.yaml").read_text())
    name_now = {info["name"] for info in new_rois.values()}
    new_cat = {}
    for group, members in src_cat.items():
        keep = []
        for m in members:
            base = m.rsplit("_", 1)[0]
            keep.append(base if base in merged else m)
        keep = [k for k in dict.fromkeys(keep) if k in name_now]
        if keep:
            new_cat[group] = keep
    (ALLEN / "roi_categories_allen26.yaml").write_text(
        "# Canonical ROI categories for Allen26 (generated by scripts/make_allen26.py)\n"
        "# Six bilateral pairs are merged; see roi_mapping_allen26.json for why.\n"
        + yaml.safe_dump(new_cat, sort_keys=False, default_flow_style=False))

    # --- colour LUT ---------------------------------------------------------
    lines = ["# Allen26 label lookup table (generated by scripts/make_allen26.py)",
             "# id  R    G    B    A  vis  msh  label"]
    for rid in sorted(mapping["rois"], key=int):
        info = mapping["rois"][rid]
        r, g, b = info.get("color_rgb", [0, 0, 0])
        a = 0 if int(rid) == 0 else 1
        lines.append(f"{rid:>4} {r:>4} {g:>4} {b:>4} {a:>4}    1    1    "
                     f'"{info["name"]}"')
    (ALLEN / "allen26_labels.txt").write_text("\n".join(lines) + "\n")

    print(f"  wrote allen26_labels.nii.gz, roi_mapping_allen26.json,")
    print(f"        roi_categories_allen26.yaml, allen26_labels.txt")
    for j, base in enumerate(merged):
        print(f"    id {2*len(lateral)+1+j:>2}  {base:<20} "
              f"(merged, |cos| = {COLLINEARITY[base]:.3f})")


if __name__ == "__main__":
    main()
