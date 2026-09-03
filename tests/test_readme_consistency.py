"""The README must agree with the code it describes.

The README was found frozen at "v1.6.0, Production Ready" on a 0.5.0 alpha
package, recommending an atlas count the registry does not have, a preset
table with the wrong number of presets and sources, a Python floor below the
one pyproject declares, a dataclass field that does not exist, and band names
the analysis CLI silently drops. Each of those is a fact the code already
records, so each can be asserted. This keeps the README from drifting again
without turning it into a generated file.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

import source_localization
from source_localization.config import ATLAS_DEFINITIONS
from source_localization.study.analysis import DEFAULT_BANDS
from source_localization.study.batch import StudyResult

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / 'README.md').read_text()
CLAUDE_MD = (ROOT / 'CLAUDE.md').read_text()
PYPROJECT = tomllib.loads((ROOT / 'pyproject.toml').read_text())
PRESET_DIR = ROOT / 'src' / 'source_localization' / 'config' / 'presets'
PRESETS = sorted(p.stem for p in PRESET_DIR.glob('*.yaml'))


def test_version_matches_package():
    assert PYPROJECT['project']['version'] == source_localization.__version__
    assert f"**Version:** {source_localization.__version__}" in README
    assert '1.6.0' not in README.split('## Changelog')[0]


def test_status_matches_classifier():
    classifiers = PYPROJECT['project']['classifiers']
    status = [c for c in classifiers if c.startswith('Development Status')][0]
    if 'Alpha' in status:
        assert '**Status:** Alpha' in README
        assert 'Production Ready' not in README.split('## Changelog')[0]


def test_python_floor_matches_pyproject():
    spec = PYPROJECT['project']['requires-python']
    floor = re.search(r'>=\s*([\d.]+)', spec).group(1)
    assert f"Python >= {floor}" in README, f"README must state Python >= {floor} (pyproject: {spec})"
    assert 'Python >= 3.8' not in README


def test_every_preset_is_documented_and_counted():
    for preset in PRESETS:
        assert f"`{preset}`" in README, f"preset {preset} missing from README"
    m = re.search(r'Available Presets \((\d+) total\)', README)
    assert m, "README preset heading must state the count"
    assert int(m.group(1)) == len(PRESETS)
    for stale in ('sphere_volumetric', 'ellipsoid_volumetric'):
        assert f"`{stale}`" not in README
        assert f"`{stale}`" not in CLAUDE_MD, f"CLAUDE.md names removed preset {stale}"


def test_atlas_table_matches_registry():
    """Each registered atlas row must carry the registry's parcel count."""
    for name, entry in ATLAS_DEFINITIONS.items():
        parcels = entry['meta']['parcels']
        row = re.search(rf'^\|\s*`{name}`\s*\|[^\n]*$', README, re.M)
        assert row, f"README atlas table lacks a row for `{name}`"
        cells = [c.strip() for c in row.group(0).strip('|').split('|')]
        assert str(parcels) in cells, (
            f"README row for {name} does not state {parcels} parcels: {row.group(0)}"
        )
    # The two numbers the old README used for the Allen atlas, which no
    # registered atlas has under the name `allen`.
    assert '49 whole-brain ROIs' not in README
    assert 'Antwerp 47-ROI' not in README


def test_study_result_fields_exist():
    fields = set(StudyResult.__dataclass_fields__)
    for attr in re.findall(r'result\.(n_\w+)', README):
        assert attr in fields, f"README uses result.{attr}, which StudyResult lacks"


def test_band_names_are_real():
    for line in re.findall(r'--bands ((?:[a-z_]+\s)+)', README):
        for band in line.split():
            assert band in DEFAULT_BANDS, f"README passes --bands {band}, not in DEFAULT_BANDS"
    assert "'gamma'" not in README, "bare 'gamma' is not a band; use low_gamma/high_gamma"


def test_citation_points_at_this_repo():
    mkdocs = yaml.safe_load((ROOT / 'mkdocs.yml').read_text())
    assert mkdocs['repo_url'] in README
    assert 'drpedapati/AlexProjects' not in README


def test_collect_output_filename_is_documented_correctly():
    # `study collect` writes subjects.csv; group_band_power.csv comes from
    # `study analyze`.
    assert 'subjects.csv' in README
