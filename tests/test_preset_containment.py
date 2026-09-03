"""Every shipped preset must build a forward that keeps every source it asked for.

A dropped source is not an error in itself (the forward step now keeps the
per-source arrays aligned when it happens), but a preset that ships with a
conductor too small for its own source space is a configuration bug: it
silently discards part of the anatomy it claims to model. This runs steps 1-5
for each preset against each registered atlas the preset can run on and
asserts zero drops.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from source_localization import Pipeline
from source_localization.config import ATLAS_DEFINITIONS
from source_localization.steps import (
    electrode_registration, bem_model, source_space, forward_solution,
)

PRESET_DIR = Path(__file__).resolve().parents[1] / 'src' / 'source_localization' / 'config' / 'presets'
PRESETS = sorted(p.stem for p in PRESET_DIR.glob('*.yaml'))
ATLASES = sorted(k for k, v in ATLAS_DEFINITIONS.items()
                 if not v.get('meta', {}).get('alias_of'))

# Presets whose conductor is known not to contain their own source space.
# Listed explicitly (and strict) so fixing one forces this table to shrink,
# and adding one cannot happen silently.
KNOWN_DROPS = {
    'roi_based_sphere': (
        'the analytical sphere (95th-percentile brain radius) leaves part of '
        'the ROI-based source space outside the inner skull on every atlas; '
        'measured 2026-09-03'
    ),
}


def _run_to_forward(preset, atlas, tmp_path):
    p = Pipeline.from_preset(preset, atlas=atlas, **{
        'outputs.dir': str(tmp_path), 'outputs.save_intermediate': False,
    })
    cfg = p.config
    prev = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for mod in (electrode_registration, bem_model, source_space, forward_solution):
            prev.update(mod.run(cfg, prev))
    return prev


@pytest.mark.slow
@pytest.mark.parametrize('atlas', ATLASES)
@pytest.mark.parametrize('preset', PRESETS)
def test_preset_forward_keeps_every_source(preset, atlas, tmp_path, request):
    if preset in KNOWN_DROPS:
        request.applymarker(pytest.mark.xfail(reason=KNOWN_DROPS[preset], strict=True))
    try:
        prev = _run_to_forward(preset, atlas, tmp_path)
    except ValueError as e:
        # The anatomical surface is an Allen product and refuses non-Allen
        # mappings by design; that refusal must be explicit, not a crash
        # somewhere downstream.
        if 'Allen32 product' in str(e):
            pytest.skip(f"{preset} does not support atlas {atlas}: {e}")
        raise
    requested = int(sum(s['nuse'] for s in prev['src']))
    kept = prev['fwd']['nsource']
    assert kept == requested, (
        f"{preset}/{atlas}: forward dropped {requested - kept} of {requested} sources"
    )
    assert len(prev['source_coords_mm']) == kept
    assert prev['n_sources'] == kept
