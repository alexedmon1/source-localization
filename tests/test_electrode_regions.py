"""The electrode region names must agree with the electrode coordinates.

Study configs once named the posterior clusters "Frontal" and the anterior
ones "Temporal" (membership right, names inverted). This checks the packaged
regions against mouse_array_coords.csv so the names cannot drift from the
geometry again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ELECTRODES = Path(__file__).resolve().parents[1] / "src/source_localization/data/electrodes"


def _load():
    coords = pd.read_csv(ELECTRODES / "mouse_array_coords.csv")
    coords = coords[coords["Label"].str.fullmatch(r"E\d+")].set_index("Label")
    regions = yaml.safe_load((ELECTRODES / "mouse_array_regions.yaml").read_text())
    return coords, regions["electrode_categories"]


def test_every_electrode_in_exactly_one_region():
    coords, regions = _load()
    members = [e for chans in regions.values() for e in chans]
    assert sorted(members) == sorted(coords.index)
    assert len(members) == len(set(members))


def test_hemisphere_matches_ml():
    coords, regions = _load()
    for name, chans in regions.items():
        ml = coords.loc[chans, "Xbregmm"].to_numpy()
        if name.startswith("Left"):
            assert (ml < 0).all(), name
        else:
            assert name.startswith("Right"), name
            assert (ml > 0).all(), name


def test_frontal_medial_posterior_order_ap():
    coords, regions = _load()
    for side in ("Left", "Right"):
        ap = {band: coords.loc[regions[f"{side} {band}"], "Ybregmm"].to_numpy()
              for band in ("Frontal", "Medial", "Posterior")}
        # bands do not overlap: every frontal site is ahead of every medial
        # site, and every medial site ahead of every posterior one
        assert ap["Frontal"].min() > ap["Medial"].max(), side
        assert ap["Medial"].min() > ap["Posterior"].max(), side
        assert np.all(ap["Frontal"] > 0) and np.all(ap["Posterior"] < -3), side
