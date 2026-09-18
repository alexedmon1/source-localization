"""The declared method vocabulary must match what the pipeline accepts.

`methods.py` exists so the CLI's choices, `source-localization list` and the docs
read from one place. That only helps if the declaration tracks the dispatchers,
so each name here is checked against the code that has to accept it — and each
dispatcher's rejection message is checked to name nothing the declaration omits.
`--method` offered three of six for several releases because the list was
written out by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from source_localization.config import ATLAS_DEFINITIONS
from source_localization.methods import (
    BEM_TYPES,
    INVERSE_METHODS,
    SAMPLING_MODES,
    SOURCE_SPACES,
    choices,
    default_of,
)

SRC = Path(__file__).resolve().parent.parent / 'src' / 'source_localization'


def _source(*parts) -> str:
    return (SRC.joinpath(*parts)).read_text()


class TestInverseMethods:
    def test_every_declared_method_has_a_branch(self):
        """Each name reaches a branch in the inverse dispatcher."""
        text = _source('steps', 'inverse_solution.py')
        for name in INVERSE_METHODS:
            upper = name.upper()
            assert (f"== '{upper}'" in text or f'"{upper}"' in text
                    or f"'{upper}'" in text), f"{name} has no branch"

    def test_dispatcher_names_no_method_the_declaration_omits(self):
        """The 'Supported: ...' error lists exactly the minimum-norm methods."""
        text = _source('steps', 'inverse_solution.py')
        m = re.search(r"Supported: ([^\"']+)", text)
        assert m, "the Unknown-method error no longer lists what it supports"
        listed = {n.strip() for n in m.group(1).split(',')}
        assert listed <= set(INVERSE_METHODS), (
            f"inverse_solution offers {listed - set(INVERSE_METHODS)}, "
            f"which methods.py does not declare")

    def test_beamformers_are_declared(self):
        """LCMV and DICS are dispatched separately, and must still be listed."""
        text = _source('steps', 'inverse_solution.py')
        assert "['LCMV', 'DICS']" in text
        assert {'LCMV', 'DICS'} <= set(INVERSE_METHODS)

    def test_beamformers_carry_their_caveat(self):
        """They are outside the mouse-scale regularization fix; say so."""
        for name in ('LCMV', 'DICS'):
            assert 'regularization' in INVERSE_METHODS[name].caveat

    def test_exactly_one_default(self):
        assert default_of(INVERSE_METHODS) == 'sLORETA'


class TestSourceSpaces:
    def test_every_declared_space_has_a_branch(self):
        text = _source('steps', 'source_space.py')
        for name in SOURCE_SPACES:
            assert f"'{name}'" in text, f"{name} is not dispatched"

    def test_dispatcher_names_no_space_the_declaration_omits(self):
        text = _source('steps', 'source_space.py')
        m = re.search(r"Valid types: ([^\"']+)", text)
        assert m, "the Unknown-source-type error no longer lists valid types"
        listed = {n.strip() for n in m.group(1).split(',')}
        assert listed == set(SOURCE_SPACES), (
            f"source_space.py and methods.py disagree: {listed ^ set(SOURCE_SPACES)}")


class TestSamplingModes:
    def test_pipeline_validates_against_the_same_two(self):
        text = _source('pipeline.py')
        m = re.search(r"Valid: ([^\"']+)", text)
        assert m, "the Unknown-source_sampling error no longer lists valid modes"
        listed = {n.strip() for n in m.group(1).split(',')}
        assert listed == set(SAMPLING_MODES)

    def test_monte_carlo_declares_that_it_is_roi_only(self):
        """The property every downstream consumer has to branch on."""
        caveat = SAMPLING_MODES['monte_carlo'].caveat
        assert 'ROI-only' in caveat and 'no vertex output' in caveat

    def test_fixed_is_the_default(self):
        assert default_of(SAMPLING_MODES) == 'fixed'


class TestBemTypes:
    def test_every_preset_names_a_declared_bem(self):
        for path in (SRC / 'config' / 'presets').glob('*.yaml'):
            cfg = yaml.safe_load(path.read_text()) or {}
            bem = (cfg.get('pipeline') or {}).get('bem_type')
            assert bem in BEM_TYPES, f"{path.name} uses undeclared BEM {bem!r}"

    def test_every_preset_names_a_declared_source_space(self):
        for path in (SRC / 'config' / 'presets').glob('*.yaml'):
            cfg = yaml.safe_load(path.read_text()) or {}
            space = (cfg.get('pipeline') or {}).get('source_type')
            assert space in SOURCE_SPACES, f"{path.name} uses undeclared space {space!r}"

    def test_every_preset_names_a_declared_inverse_method(self):
        for path in (SRC / 'config' / 'presets').glob('*.yaml'):
            cfg = yaml.safe_load(path.read_text()) or {}
            method = (cfg.get('inverse') or {}).get('method')
            assert method in INVERSE_METHODS, f"{path.name} uses undeclared {method!r}"


class TestCliWiring:
    def test_method_choices_come_from_the_declaration(self):
        """Not a hand-written subset — the bug this module exists to prevent."""
        from source_localization.cli import _create_run_parser

        parser = _create_run_parser()
        action = next(a for a in parser._actions if a.dest == 'method')
        assert action.choices == choices(INVERSE_METHODS)

    def test_source_sampling_is_reachable_from_the_cli(self):
        from source_localization.cli import _create_run_parser

        parser = _create_run_parser()
        action = next(a for a in parser._actions if a.dest == 'source_sampling')
        assert action.choices == choices(SAMPLING_MODES)

    def test_atlas_choices_come_from_the_registry(self):
        from source_localization.cli import _create_run_parser

        parser = _create_run_parser()
        action = next(a for a in parser._actions if a.dest == 'atlas')
        assert set(action.choices) == set(ATLAS_DEFINITIONS)


class TestListCommand:
    @pytest.mark.parametrize('flag', ['presets', 'methods', 'atlases'])
    def test_each_section_prints(self, capsys, flag):
        from source_localization.cli import _run_list_command

        args = type('A', (), {'presets': False, 'methods': False, 'atlases': False})()
        setattr(args, flag, True)
        assert _run_list_command(args) == 0
        assert capsys.readouterr().out.strip()

    def test_listing_covers_every_preset(self, capsys):
        from source_localization.cli import _run_list_command

        args = type('A', (), {'presets': True, 'methods': False, 'atlases': False})()
        _run_list_command(args)
        out = capsys.readouterr().out
        for path in (SRC / 'config' / 'presets').glob('*.yaml'):
            assert path.stem in out

    def test_listing_covers_every_atlas_and_method(self, capsys):
        from source_localization.cli import _run_list_command

        args = type('A', (), {'presets': False, 'methods': True, 'atlases': True})()
        _run_list_command(args)
        out = capsys.readouterr().out
        for name in list(ATLAS_DEFINITIONS) + list(INVERSE_METHODS) + list(SAMPLING_MODES):
            assert name in out, f"{name} is selectable but not listed"
