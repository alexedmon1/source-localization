"""Configuration management for source localization pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def _load_atlas_registry() -> Dict[str, Any]:
    registry_file = Path(__file__).parent / 'data' / 'atlas' / 'registry.yaml'
    with open(registry_file) as f:
        return yaml.safe_load(f)


# Atlas definitions: maps atlas name -> {'inputs': {...}, 'meta': {...}}
# All paths are relative to the package data directory
# To add a new atlas, edit data/atlas/registry.yaml
ATLAS_DEFINITIONS = _load_atlas_registry()

# Default atlas used by all presets
DEFAULT_ATLAS = 'antwerp'

# The validation CLI grew its own atlas vocabulary before the registry existed
# ('full', 'coarse_22roi'), and scripts/ still pass those names. They are mapped
# here rather than in either CLI so there is exactly one place where a name is
# resolved, and so both entry points accept both vocabularies.
LEGACY_ATLAS_ALIASES = {
    'full': 'antwerp',
    'coarse_22roi': 'coarse22',
}


def resolve_atlas_name(name: str) -> str:
    """Map an atlas name (possibly a legacy alias) to its registry key.

    Raises
    ------
    ValueError
        If the name is neither a registry key nor a known alias.
    """
    if name in ATLAS_DEFINITIONS:
        return name
    if name in LEGACY_ATLAS_ALIASES:
        return LEGACY_ATLAS_ALIASES[name]
    available = ', '.join(sorted(ATLAS_DEFINITIONS))
    aliases = ', '.join(sorted(LEGACY_ATLAS_ALIASES))
    raise ValueError(
        f"Unknown atlas: {name}. Available: {available} "
        f"(legacy aliases: {aliases})")


def atlas_input_paths(name: str) -> Dict[str, str]:
    """The ``inputs:`` path mapping for an atlas, resolving aliases."""
    entry = ATLAS_DEFINITIONS[resolve_atlas_name(name)]
    return dict(entry.get('inputs', entry))


class Config:
    """Pipeline configuration manager."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._validate()

    @classmethod
    def from_preset(cls, preset_name: str) -> 'Config':
        """Load configuration from preset.

        Parameters
        ----------
        preset_name : str
            Name of preset (e.g., 'sphere_volumetric')

        Returns
        -------
        config : Config
            Loaded configuration
        """
        preset_file = Path(__file__).parent / 'config' / 'presets' / f'{preset_name}.yaml'

        if not preset_file.exists():
            raise ValueError(f"Preset not found: {preset_name}")

        with open(preset_file) as f:
            config = yaml.safe_load(f)

        config.setdefault('provenance', {}).update(
            {'config_source': 'preset', 'preset': preset_name}
        )

        return cls(config)

    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """Load configuration from file.

        Parameters
        ----------
        config_file : str
            Path to configuration YAML file

        Returns
        -------
        config : Config
            Loaded configuration
        """
        with open(config_file) as f:
            config = yaml.safe_load(f)

        config.setdefault('provenance', {}).update(
            {'config_source': 'file', 'config_file': str(config_file)}
        )

        return cls(config)

    @classmethod
    def from_bem_source(cls, bem_type: str, source_type: str, **overrides) -> 'Config':
        """Build configuration from BEM and source type.

        Parameters
        ----------
        bem_type : str
            BEM type ('sphere', 'ellipsoid', 'tissue')
        source_type : str
            Source space type ('volumetric', 'surface')
        **overrides : dict
            Additional configuration overrides (use dot notation for nested keys)

        Returns
        -------
        config : Config
            Built configuration
        """
        # Load default config
        default_file = Path(__file__).parent / 'config' / 'default_config.yaml'
        with open(default_file) as f:
            config = yaml.safe_load(f)

        # Set BEM and source type
        config['pipeline']['bem_type'] = bem_type
        config['pipeline']['source_type'] = source_type

        # Apply overrides
        for key, value in overrides.items():
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

        config.setdefault('provenance', {}).update(
            {'config_source': 'bem_source'}
        )

        return cls(config)

    def apply_atlas(self, atlas_name: str) -> None:
        """Override atlas input paths for a named atlas.

        Only the registry entry's ``inputs:`` block reaches the config. Its
        ``meta:`` block is descriptive (parcel counts, coverage, tier scheme) and
        must not be copied — an earlier version copied every key, which put a
        ``full_brain_coverage`` boolean into ``inputs`` alongside the file paths.

        Parameters
        ----------
        atlas_name : str
            Atlas name; see :data:`ATLAS_DEFINITIONS` for available options.
            Currently 'antwerp', 'allen', 'allen32', 'allen64'.
        """
        # Registry entries are {'inputs': {...}, 'meta': {...}}; the older flat
        # form, where the entry was the path mapping itself, still works.
        atlas_paths = atlas_input_paths(atlas_name)

        if 'inputs' not in self._config:
            self._config['inputs'] = {}
        for key, path in atlas_paths.items():
            if key == 'meta':
                continue
            self._config['inputs'][key] = path

        # The atlas is applied over whatever the preset declared, so the
        # `inputs:` paths alone no longer say which atlas was asked for by name.
        self._config.setdefault('provenance', {})['atlas'] = atlas_name

    @staticmethod
    def atlas_meta(atlas_name: str) -> Dict[str, Any]:
        """Descriptive metadata for a registered atlas (parcel count, coverage).

        Returns an empty dict for an entry that declares no ``meta:`` block.
        """
        entry = ATLAS_DEFINITIONS[resolve_atlas_name(atlas_name)]
        return dict(entry.get('meta', {}))

    def _validate(self):
        """Validate configuration."""
        required = ['pipeline', 'bem', 'source_space', 'inverse']
        for key in required:
            if key not in self._config:
                raise ValueError(f"Missing required config section: {key}")

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
