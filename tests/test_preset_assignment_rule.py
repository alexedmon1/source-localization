"""Every preset must agree on how a source gets its ROI label.

Why this test exists
--------------------
`roi.use_proximity` assigns a source to every parcel within `proximity_radius_mm`
instead of to the voxel it lands in. It defaulted to **true** in the Cartesian and
surface presets and **false** in the ROI-based and shell presets, and no study
config overrode it. A stability sweep that varied only the preset and the inverse
operator therefore varied the assignment rule as well, silently, and the affected
runs had to be repeated (FORGE WP-13 -> WP-14, 2026-08-05).

The defect was never the feature. It was that two presets a study treats as
interchangeable disagreed on a default that changes the result. So the invariant
worth pinning is *agreement*, not any particular value — plus the specific value
we settled on, since the whole point is that nearest-voxel is now the rule.

The feature is scheduled for removal; when it goes, delete this file with it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "src/source_localization/config"
PRESETS = sorted((CONFIG / "presets").glob("*.yaml"))
ALL_CONFIGS = [CONFIG / "default_config.yaml", *PRESETS]


def _roi_block(path: Path) -> dict:
    return yaml.safe_load(path.read_text()).get("roi", {}) or {}


def test_presets_exist():
    """Guard against the glob silently matching nothing."""
    assert len(PRESETS) >= 6, f"expected the preset set, found {[p.name for p in PRESETS]}"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_assignment_rule_is_nearest_voxel(path: Path):
    """No shipped config may turn proximity assignment on.

    A study that overrides it explicitly is making a deliberate choice and is
    free to; what must not happen again is inheriting it without asking.
    """
    roi = _roi_block(path)
    assert roi.get("use_proximity", False) is False, (
        f"{path.name} enables proximity assignment. Sources must be assigned to "
        f"the parcel they land in, and every preset must agree — see the module "
        f"docstring for what disagreement cost."
    )


def test_all_presets_agree():
    """Stated as agreement, so this still fails if someone flips them all back."""
    rules = {p.name: _roi_block(p).get("use_proximity", False) for p in PRESETS}
    assert len(set(rules.values())) == 1, (
        f"presets disagree on the ROI assignment rule: {rules}"
    )


def test_code_default_matches_configs():
    """An absent key must mean nearest-voxel, not proximity.

    Checked against the source rather than by running the step, which needs a
    fitted inverse solution; the default lives in one `.get()` call.
    """
    src = (
        CONFIG.parent / "steps/roi_extraction.py"
    ).read_text()
    assert "config['roi'].get('use_proximity', False)" in src, (
        "roi_extraction.py must default use_proximity to False when the key is "
        "absent, so a config that omits it cannot silently get proximity."
    )
