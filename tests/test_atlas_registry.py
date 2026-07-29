"""Tests for the atlas registry and atlas selection.

These pin properties that were each broken at some point and were invisible
because nothing checked them:

- ``--atlas`` accepted a hardcoded list that omitted registered atlases, so
  ``allen32``/``allen64`` could not be selected from the CLI at all.
- The CLI help described ``allen`` as having 49 ROIs when it has 32.
- ``apply_atlas`` copied every key of a registry entry into ``inputs``, so a
  ``full_brain_coverage`` boolean landed among the file paths.
- That flag claimed full brain coverage for the Allen atlases, which label
  roughly two thirds of the brain mask.

The coverage numbers are asserted loosely against the shipped label volumes, so
the registry cannot drift away from the files it describes.
"""
import numpy as np
import pytest

from source_localization.config import ATLAS_DEFINITIONS, Config

PACKAGE_DATA = None  # resolved lazily in _package_dir()

EXPECTED_PARCELS = {'antwerp': 46, 'allen': 32, 'allen32': 32, 'allen64': 64,
                    'coarse22': 22}


def _package_dir():
    import source_localization
    from pathlib import Path
    return Path(source_localization.__file__).parent


def test_registry_exposes_the_expected_atlases():
    assert set(ATLAS_DEFINITIONS) == set(EXPECTED_PARCELS)


@pytest.mark.parametrize('name', sorted(EXPECTED_PARCELS))
def test_every_registry_path_resolves(name):
    base = _package_dir()
    for key, rel in ATLAS_DEFINITIONS[name]['inputs'].items():
        assert (base / rel).exists(), f"{name}.{key} -> missing file {rel}"


@pytest.mark.parametrize('name', sorted(EXPECTED_PARCELS))
def test_apply_atlas_copies_paths_and_not_metadata(name):
    """``meta:`` is descriptive and must never reach the pipeline's inputs."""
    config = Config.from_preset('shell_ellipsoid')
    config.apply_atlas(name)
    inputs = config._config['inputs']

    assert 'meta' not in inputs
    assert 'full_brain_coverage' not in inputs
    for key, rel in ATLAS_DEFINITIONS[name]['inputs'].items():
        assert inputs[key] == rel


def test_unknown_atlas_names_the_available_ones():
    config = Config.from_preset('shell_ellipsoid')
    with pytest.raises(ValueError, match='Unknown atlas'):
        config.apply_atlas('not_an_atlas')


@pytest.mark.parametrize('name,expected', sorted(EXPECTED_PARCELS.items()))
def test_declared_parcel_count_matches_the_label_volume(name, expected):
    """The registry's parcel count must match what the NIfTI actually holds."""
    nib = pytest.importorskip('nibabel')
    path = _package_dir() / ATLAS_DEFINITIONS[name]['inputs']['brain_labels']
    labels = np.asarray(nib.load(str(path)).get_fdata()).astype(int)
    n_parcels = len(np.unique(labels)) - 1          # exclude background 0

    assert n_parcels == expected
    assert ATLAS_DEFINITIONS[name]['meta']['parcels'] == expected


@pytest.mark.parametrize('name', sorted(EXPECTED_PARCELS))
def test_declared_coverage_matches_the_brain_mask(name):
    """No atlas covers the whole brain mask, and the declared figure is honest.

    Guards the specific claim that replaced ``full_brain_coverage: true``.
    """
    nib = pytest.importorskip('nibabel')
    base = _package_dir()
    entry = ATLAS_DEFINITIONS[name]

    mask = np.asarray(nib.load(
        str(base / entry['inputs']['brain_mask'])).get_fdata()) > 0
    labels = np.asarray(nib.load(
        str(base / entry['inputs']['brain_labels'])).get_fdata()).astype(int)

    measured = 100.0 * (labels[mask] > 0).mean()
    declared = entry['meta']['brain_mask_coverage_pct']

    assert measured == pytest.approx(declared, abs=0.5)
    assert measured < 100.0, "no shipped atlas tiles the entire brain mask"


def test_allen_is_an_alias_of_allen32():
    assert (ATLAS_DEFINITIONS['allen']['inputs']
            == ATLAS_DEFINITIONS['allen32']['inputs'])
    assert ATLAS_DEFINITIONS['allen']['meta']['alias_of'] == 'allen32'


def test_atlas_meta_is_readable_and_isolated():
    meta = Config.atlas_meta('allen32')
    assert meta['parcels'] == 32
    assert meta['parcels_per_hemisphere'] == 16
    meta['parcels'] = -1                       # returned copy must not alias
    assert Config.atlas_meta('allen32')['parcels'] == 32


def test_cli_offers_every_registered_atlas():
    """The regression that made allen32/allen64 unreachable from the CLI."""
    from source_localization.cli import _create_run_parser

    parser = _create_run_parser()
    action = next(a for a in parser._actions if '--atlas' in a.option_strings)

    assert set(action.choices) == set(ATLAS_DEFINITIONS)


def test_cli_atlas_help_reports_true_parcel_counts():
    """The old help string said 'allen (49 ROIs)'; allen has 32."""
    from source_localization.cli import _atlas_choices_help

    text = _atlas_choices_help()
    assert 'allen32 (32 parcels)' in text
    assert 'allen64 (64 parcels)' in text
    assert 'antwerp (46 parcels)' in text
    assert '49' not in text


# --------------------------------------------------------------- alias layer

@pytest.mark.parametrize('alias,canonical', [
    ('full', 'antwerp'),
    ('coarse_22roi', 'coarse22'),
    ('allen32', 'allen32'),        # registry keys resolve to themselves
    ('antwerp', 'antwerp'),
])
def test_legacy_aliases_resolve(alias, canonical):
    from source_localization.config import resolve_atlas_name
    assert resolve_atlas_name(alias) == canonical


def test_unknown_name_lists_both_vocabularies():
    from source_localization.config import resolve_atlas_name
    with pytest.raises(ValueError) as excinfo:
        resolve_atlas_name('not_an_atlas')
    message = str(excinfo.value)
    assert 'coarse22' in message and 'full' in message


def test_both_clis_accept_every_registered_atlas():
    """The two CLIs had disjoint atlas vocabularies overlapping on one name."""
    from source_localization.cli import _create_run_parser
    from source_localization.validation.cli import create_parser

    def choices(parser):
        action = next(a for a in parser._actions
                      if '--atlas' in a.option_strings)
        return set(action.choices)

    localization = choices(_create_run_parser())
    validation = choices(create_parser())

    assert set(ATLAS_DEFINITIONS) <= localization
    assert set(ATLAS_DEFINITIONS) <= validation
    # Validation additionally keeps the legacy names its scripts still pass.
    assert {'full', 'coarse_22roi'} <= validation


# ------------------------------------------------- the affine-convention bug

@pytest.mark.parametrize('name', sorted(EXPECTED_PARCELS))
def test_label_volume_lands_on_the_brain_in_mm(name):
    """Every atlas must place its labels where sources actually are.

    ``roi_extraction`` used the raw NIfTI affine, which is right for the
    true-unit label volumes but wrong for ``coarse22``, whose header is
    10x-inflated: its labels landed at +/-49 mm while sources sit at +/-5 mm.
    Nothing raised, because the nearest-labeled-voxel fallback always finds a
    voxel — it just collapsed all 215 shell sources onto 2 of the 22 ROIs.
    """
    nib = pytest.importorskip('nibabel')
    from source_localization.utils.atlas import get_true_affine

    path = _package_dir() / ATLAS_DEFINITIONS[name]['inputs']['brain_labels']
    nii = nib.load(str(path))
    labels = np.asarray(nii.get_fdata()).astype(int)

    occupied = np.argwhere(labels > 0)
    corners = np.array([occupied.min(axis=0), occupied.max(axis=0)])
    mm = nib.affines.apply_affine(get_true_affine(nii), corners)

    # The mouse brain spans roughly +/-5 x +/-8 x +/-3 mm in this space.
    assert np.abs(mm).max() < 15.0, (
        f"{name} labels span {np.abs(mm).max():.1f} mm — the affine convention "
        f"is being mishandled")


@pytest.mark.parametrize('name', ['antwerp', 'allen32', 'allen64'])
def test_get_true_affine_is_a_noop_for_true_unit_labels(name):
    """The roi_extraction fix must not change the atlases already in use."""
    nib = pytest.importorskip('nibabel')
    from source_localization.utils.atlas import get_true_affine

    path = _package_dir() / ATLAS_DEFINITIONS[name]['inputs']['brain_labels']
    nii = nib.load(str(path))
    assert np.allclose(nii.affine, get_true_affine(nii))


def test_coarse22_needs_the_correction():
    """Complement of the above: coarse22 is the one that actually changes."""
    nib = pytest.importorskip('nibabel')
    from source_localization.utils.atlas import get_true_affine

    path = _package_dir() / ATLAS_DEFINITIONS['coarse22']['inputs']['brain_labels']
    nii = nib.load(str(path))
    assert not np.allclose(nii.affine, get_true_affine(nii))


def test_sources_spread_across_rois_under_every_atlas():
    """End-to-end guard on the collapse: ROI assignment must not degenerate.

    Uses a synthetic source cloud spanning the brain rather than a pipeline
    artifact, so the test is self-contained.
    """
    nib = pytest.importorskip('nibabel')
    import json as _json
    from source_localization.steps.roi_extraction import map_sources_to_rois_nearest
    from source_localization.utils.atlas import get_true_affine

    rng = np.random.default_rng(0)
    sources = rng.uniform([-4, -6, -2.5], [4, 6, 2.5], size=(200, 3))

    for name in sorted(EXPECTED_PARCELS):
        entry = ATLAS_DEFINITIONS[name]['inputs']
        nii = nib.load(str(_package_dir() / entry['brain_labels']))
        labels = np.asarray(nii.get_fdata()).astype(int)
        mapping = _json.load(open(_package_dir() / entry['roi_mapping']))['rois']
        label_to_roi = {int(k): v['name'] for k, v in mapping.items()
                        if int(k) > 0}

        assigned = map_sources_to_rois_nearest(
            sources, labels, get_true_affine(nii), label_to_roi)
        populated = sum(1 for v in assigned.values() if len(v) > 0)

        assert populated > EXPECTED_PARCELS[name] // 2, (
            f"{name}: only {populated}/{EXPECTED_PARCELS[name]} ROIs received "
            f"any of 200 spread-out sources")
