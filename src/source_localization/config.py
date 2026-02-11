"""Configuration management for source localization pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


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

        return cls(config)

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
