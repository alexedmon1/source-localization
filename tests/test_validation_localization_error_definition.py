"""Lock the localization-error reference convention.

Regression guard for a defect that reached publication: releases up to v0.3.0
computed localization error from the SNAPPED source position but reported it
under the generic key ``localization_errors``. That contradicted both
``simulate_dipole()``'s documented metadata contract and the published MS1
benchmark (whose Table 2 caption states the test grid is "independent of the
source space, ensuring true spatial estimation rather than grid recovery" --
i.e. the REQUESTED-position convention).

Because only one error was persisted, past runs could not be re-derived under
the other convention without re-simulating, and the discrepancy was not
discoverable from the outputs alone.

These tests fail if:
  - the primary convention silently flips back to 'snapped';
  - either error series stops being persisted;
  - the raw positions needed to re-derive either convention are dropped;
  - two-dipole metadata loses the requested separation (the design variable of
    a resolvability sweep).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from source_localization.validation import runner as runner_mod
from source_localization.validation import simulation as simulation_mod


# --------------------------------------------------------------------------
# The convention itself
# --------------------------------------------------------------------------

def test_primary_definition_is_requested():
    """The primary reported error must be measured from the REQUESTED position."""
    assert runner_mod.LOCALIZATION_ERROR_DEFINITION == 'requested', (
        "Primary localization-error convention changed. The published benchmark "
        "and simulate_dipole()'s contract both specify 'requested'. If this is "
        "an intentional change, update the manuscript, the Table 2 caption, and "
        "this test together -- never silently."
    )


@pytest.mark.parametrize("loop_name", ["run", "_run_single_mode"])
def test_primary_error_uses_requested_position_in_every_loop(loop_name):
    """Both simulation loops must compute the primary error from `requested`.

    Guards the specific historical failure: one loop was fixed and the other
    left on the old convention, so results depended on which entry point ran.
    """
    fn = getattr(runner_mod.ValidationRunner, loop_name, None)
    if fn is None:
        pytest.skip(f"{loop_name} not present in this version")
    src = inspect.getsource(fn)

    if 'compute_localization_error' not in src:
        pytest.skip(f"{loop_name} does not compute localization error")

    # The primary assignment must reference the requested position.
    assert 'loc_error = compute_localization_error(' in src.replace('\n', ' ').replace('  ', ' ') or \
           'loc_error = compute_localization_error(' in src, (
        f"{loop_name}: could not find the primary `loc_error` assignment"
    )
    primary = src.split('loc_error = compute_localization_error(')[1][:200]
    assert 'requested_position_mm' in primary, (
        f"{loop_name}: primary localization error is NOT measured from "
        f"requested_position_mm. Got: {primary.strip()[:120]!r}"
    )

    # The diagnostic must still exist alongside it.
    assert 'loc_error_snapped' in src, (
        f"{loop_name}: the snapped diagnostic was removed. Keep it -- it is how "
        "the requested/snapped gap is explained and how historical runs are "
        "reconciled."
    )


# --------------------------------------------------------------------------
# Persistence: both series and the positions needed to re-derive either
# --------------------------------------------------------------------------

REQUIRED_KEYS = [
    'localization_errors',           # primary (requested)
    'localization_errors_snapped',   # diagnostic
    'snapping_errors',
    'true_positions_mm',             # requested
    'snapped_positions_mm',
    'estimated_positions_mm',        # without this, neither can be re-derived
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_runner_initialises_required_raw_arrays(key):
    """Every quantity needed to re-derive either convention must be persisted."""
    src = inspect.getsource(runner_mod)
    assert f"'{key}': []" in src, (
        f"raw_data key '{key}' is no longer initialised. Dropping it makes past "
        "runs non-reconcilable without re-simulating -- exactly the situation "
        "this guard exists to prevent."
    )


def test_definition_is_stamped_into_results():
    """Outputs must self-document which convention produced them."""
    src = inspect.getsource(runner_mod)
    assert "'localization_error_definition': LOCALIZATION_ERROR_DEFINITION" in src, (
        "metrics.json no longer records localization_error_definition. Without "
        "it, a results file cannot be interpreted without knowing the code SHA."
    )


# --------------------------------------------------------------------------
# Two-dipole path (MS2 one-source / two-source validation)
# --------------------------------------------------------------------------

def test_two_dipole_metadata_records_requested_separation():
    """Resolvability sweeps vary REQUESTED separation; it must be recorded.

    Snapping perturbs the pair, so achieved separation != requested separation.
    Reporting only the achieved value loses the sweep's independent variable.
    """
    src = inspect.getsource(simulation_mod.DipoleSimulator.simulate_two_dipoles)
    for key in ('requested_separation_mm', 'separation_mm'):
        assert f"'{key}'" in src, f"two-dipole metadata lost '{key}'"
    assert 'snapped_source_position_mm' in src, (
        "achieved separation must be computed from the snapped positions "
        "(what was actually simulated), not a legacy alias"
    )


def test_legacy_aliases_still_map_as_documented():
    """`actual_position_mm` is the SNAPPED position; guard the alias meaning.

    Three call sites once used this alias believing it was ground truth. Keep
    its meaning pinned so the same mistake cannot be made silently again.
    """
    src = inspect.getsource(simulation_mod.DipoleSimulator.simulate_dipole)
    assert "'actual_position_mm': actual_position_mm" in src
    assert "'snapped_source_position_mm': actual_position_mm" in src, (
        "actual_position_mm must remain an alias for the SNAPPED position"
    )


def test_snapping_error_is_consistent_with_positions():
    """snapping_error_mm must equal |requested - snapped| (self-consistency)."""
    src = inspect.getsource(simulation_mod.DipoleSimulator.simulate_dipole)
    assert "'snapping_error_mm': np.linalg.norm(position_mm - actual_position_mm)" in src


# --------------------------------------------------------------------------
# Numeric sanity of the convention (no simulation required)
# --------------------------------------------------------------------------

def test_requested_error_is_never_less_than_snapped_minus_snapping():
    """Triangle inequality ties the three recorded quantities together.

    |requested - peak| <= |requested - snapped| + |snapped - peak|
    A future refactor that mixes conventions would break this relation.
    """
    from source_localization.validation.metrics import compute_localization_error

    rng = np.random.RandomState(0)
    for _ in range(200):
        requested, snapped, peak = rng.randn(3, 3) * 5
        e_req = compute_localization_error(requested, peak)
        e_snap = compute_localization_error(snapped, peak)
        snapping = compute_localization_error(requested, snapped)
        assert e_req <= e_snap + snapping + 1e-9
        assert e_snap <= e_req + snapping + 1e-9
