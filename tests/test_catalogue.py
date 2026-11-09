"""The model catalogue must identify every model properly and stay documented.

``methods.py`` declares each location posterior, prior and calibration with a
stable id and a version (bayes-sl reports/2026-11-09_field_method_library_plan.md,
"Identification and documentation"). These tests keep that declaration honest:
ids unique and well formed, every prior a posterior rests on catalogued, every
implementation importable, the generated page current, and the page reachable
from the README and the docs site.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import yaml

from source_localization import methods as M

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'docs' / 'methods' / 'CATALOGUE.md'
ENTRIES = M.catalogue_entries()
ID = re.compile(r'^(locate|prior|calibration|field|whiten)\.[a-z0-9_]+$')


def test_ids_are_unique_well_formed_and_keyed_by_themselves():
    seen = []
    for section in M.CATALOGUE.values():
        for key, m in section.items():
            assert key == m.id, (key, m.id)
            assert ID.match(m.id), m.id
            seen.append(m.id)
    assert len(seen) == len(set(seen))


def test_versions_are_positive_integers():
    for m in ENTRIES.values():
        assert isinstance(m.version, int) and m.version >= 1, m.id


def test_every_entry_says_what_it_is_and_how_it_is_made():
    for m in ENTRIES.values():
        assert m.summary and m.model and m.implementation, m.id


@pytest.mark.parametrize('m', list(M.LOCATION_POSTERIORS.values()), ids=lambda m: m.id)
def test_posteriors_name_their_priors_and_each_is_catalogued(m):
    assert m.priors, m.id
    for p in m.priors:
        assert p in M.LOCATION_PRIORS, f'{m.id} rests on {p!r}, which is not a catalogued prior'


@pytest.mark.parametrize('m', list(ENTRIES.values()), ids=lambda m: m.id)
def test_implementation_imports_or_is_marked_external(m):
    if m.implementation.startswith('external:'):
        assert 'phase' in m.implementation, f'{m.id}: say which phase moves it here'
        return
    module, _, attr = m.implementation.partition(':')
    obj = importlib.import_module(module)
    for part in attr.split('.'):
        obj = getattr(obj, part)


@pytest.mark.parametrize('m', list(ENTRIES.values()), ids=lambda m: m.id)
def test_named_test_files_exist(m):
    for f in filter(None, (t.strip() for t in m.tests.split(','))):
        assert (ROOT / f).exists(), f'{m.id}: {f}'


def test_calibrations_state_their_domain():
    """Every measured calibration setting names its domain (calibration.none has none; calibration.procedure is how
    settings are measured, not a setting)."""
    for m in M.MAP_CALIBRATIONS.values():
        if m.id not in ('calibration.none', 'calibration.procedure'):
            assert 'MEA30' in m.validity, m.id


def test_generated_page_is_current():
    assert PAGE.exists(), 'run: python -m source_localization.methods > docs/methods/CATALOGUE.md'
    assert PAGE.read_text() == M.catalogue_markdown(), (
        'docs/methods/CATALOGUE.md is stale; run: '
        'python -m source_localization.methods > docs/methods/CATALOGUE.md')


def test_page_is_linked_from_readme_and_docs_site():
    assert 'docs/methods/CATALOGUE.md' in (ROOT / 'README.md').read_text()
    nav = yaml.safe_load((ROOT / 'mkdocs.yml').read_text())['nav']
    assert 'methods/CATALOGUE.md' in repr(nav)
