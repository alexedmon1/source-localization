"""Every bundled atlas file must ship: an editable install hides a package-data pattern that misses a directory."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "source_localization"


def test_package_data_matches_every_atlas_file():
    tomllib = pytest.importorskip("tomllib")
    with open(ROOT / "pyproject.toml", "rb") as f:
        patterns = tomllib.load(f)["tool"]["setuptools"]["package-data"]["source_localization"]
    matched = {p for pattern in patterns for p in PACKAGE.glob(pattern) if p.is_file()}
    atlas = [p for p in (PACKAGE / "data" / "atlas").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert any(p.parent.name == "allen" for p in atlas)
    assert not [str(p) for p in atlas if p not in matched]
