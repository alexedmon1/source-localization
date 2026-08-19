"""Main pipeline orchestration."""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml

from .config import Config
from .steps import (
    electrode_registration,
    eeg_data,
    bem_model,
    source_space,
    forward_solution,
    inverse_solution,
    roi_extraction,
    spectral_analysis,
    visualization
)


class Pipeline:
    """Source localization pipeline orchestrator.

    The core pipeline has 7 steps that produce MNE-compatible .set files:
    1. Electrode Registration
    2. EEG Data Loading
    3. BEM Model
    4. Source Space
    5. Forward Solution
    6. Inverse Solution
    7. ROI Extraction (exports roi_timeseries_magnitude.set and roi_timeseries_signed.set)

    Optional post-processing (call separately):
    - run_spectral_analysis(): Compute band power
    - run_visualization(): Generate plots
    """

    # Core pipeline steps (produce .set export)
    STEPS = [
        ('electrode_registration', electrode_registration),
        ('eeg_data', eeg_data),
        ('bem_model', bem_model),
        ('source_space', source_space),
        ('forward_solution', forward_solution),
        ('inverse_solution', inverse_solution),
        ('roi_extraction', roi_extraction),
    ]

    # Optional post-processing steps
    OPTIONAL_STEPS = [
        ('spectral_analysis', spectral_analysis),
        ('visualization', visualization),
    ]

    def __init__(self, config: Config):
        """Initialize pipeline with configuration.

        Parameters
        ----------
        config : Config
            Pipeline configuration
        """
        self.config = config
        self.outputs = {}
        self.step_outputs = {}

    @classmethod
    def from_preset(cls, preset_name: str, atlas: str = None, **overrides) -> 'Pipeline':
        """Create pipeline from preset configuration.

        Parameters
        ----------
        preset_name : str
            Name of preset (e.g., 'sphere_volumetric')
        atlas : str, optional
            Atlas to use ('antwerp', 'allen'). If None, uses
            the preset's default atlas paths.
        **overrides : dict
            Configuration overrides (use dot notation for nested keys)

        Returns
        -------
        pipeline : Pipeline
            Configured pipeline instance
        """
        config = Config.from_preset(preset_name)

        # Apply atlas override
        if atlas is not None:
            config.apply_atlas(atlas)

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                # Handle nested keys (e.g., 'inverse.snr')
                keys = key.split('.')
                d = config._config
                for k in keys[:-1]:
                    d = d[k]
                d[keys[-1]] = value

        return cls(config)

    @classmethod
    def from_bem_source(cls, bem_type: str, source_type: str, **kwargs) -> 'Pipeline':
        """Create pipeline from BEM and source type.

        Parameters
        ----------
        bem_type : str
            BEM type ('sphere', 'ellipsoid', 'tissue')
        source_type : str
            Source space type ('volumetric', 'surface')
        **kwargs : dict
            Additional configuration overrides

        Returns
        -------
        pipeline : Pipeline
            Configured pipeline instance
        """
        config = Config.from_bem_source(bem_type, source_type, **kwargs)
        return cls(config)

    @classmethod
    def from_file(cls, config_file: str) -> 'Pipeline':
        """Create pipeline from configuration file.

        Parameters
        ----------
        config_file : str
            Path to configuration YAML file

        Returns
        -------
        pipeline : Pipeline
            Configured pipeline instance
        """
        config = Config.from_file(config_file)
        return cls(config)

    @staticmethod
    def _plain(value):
        """Make a config value YAML-safe: Paths and tuples become str and list."""
        if isinstance(value, dict):
            return {str(k): Pipeline._plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Pipeline._plain(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    def _write_resolved_config(self, output_path: Path) -> Path:
        """Snapshot the fully resolved config next to the outputs it produced.

        The HTML report shows the settings that decide a run, but a report is
        rendered for people. This file is the machine-readable record: preset,
        atlas, source-space method and spacing, orientation, every override the
        CLI applied. Without it a finished output directory cannot say what
        built it, and reruns cannot be shown to be reruns.
        """
        # Imported here, not at module scope: `__init__` imports this module, so
        # a top-level `from . import __version__` is a circular import.
        from . import __version__

        snapshot = {
            'source_localization_version': __version__,
            'written': datetime.now().isoformat(timespec='seconds'),
            'config': self._plain(self.config.to_dict()),
        }
        target = output_path / 'data' / 'config_resolved.yaml'
        with open(target, 'w') as f:
            yaml.safe_dump(snapshot, f, sort_keys=False, default_flow_style=False)
        return target

    def run(self, eeg_file: Optional[str] = None, output_dir: Optional[str] = None,
            include_spectral: bool = False, include_visualization: bool = False,
            skip_roi_extraction: bool = False) -> Dict[str, Any]:
        """
        Run the core source localization pipeline.

        The core pipeline produces MNE-compatible .set files with ROI time series.
        Spectral analysis and visualization are optional post-processing steps.

        Parameters
        ----------
        eeg_file : str, optional
            Path to EEG file (overrides config)
        output_dir : str, optional
            Output directory (overrides config)
        include_spectral : bool, default=False
            If True, also run spectral analysis after core pipeline
        include_visualization : bool, default=False
            If True, also run visualization after core pipeline
        skip_roi_extraction : bool, default=False
            If True, stop after step 6 (inverse solution) and skip ROI
            extraction.  ROI extraction can then be performed on-the-fly
            by source-analytics.

        Returns
        -------
        results : dict
            Pipeline results with outputs from all steps
        """
        # Override config if provided
        if eeg_file:
            self.config._config['inputs']['eeg_file'] = eeg_file
        if output_dir:
            self.config._config['outputs']['dir'] = Path(output_dir)

        # Create output directory structure
        output_path = Path(self.config['outputs']['dir'])
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'data').mkdir(exist_ok=True)
        (output_path / 'figures').mkdir(exist_ok=True)

        self._write_resolved_config(output_path)

        print("="*80)
        print(f"SOURCE LOCALIZATION PIPELINE: {self.config['pipeline']['name']}")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"BEM Type: {self.config['pipeline']['bem_type']}")
        print(f"Source Type: {self.config['pipeline']['source_type']}")
        print(f"Inverse Method: {self.config['inverse']['method']}")
        print(f"Output: {output_path}")
        print("="*80)

        # Determine which steps to run
        steps_to_run = self.STEPS
        if skip_roi_extraction:
            steps_to_run = [
                (name, mod) for name, mod in self.STEPS
                if name != 'roi_extraction'
            ]

        # Run core pipeline steps
        previous_outputs = {}

        for i, (step_name, step_module) in enumerate(steps_to_run, 1):
            print(f"\n{'='*80}")
            print(f"STEP {i}/{len(steps_to_run)}: {step_name.replace('_', ' ').title()}")
            print(f"{'='*80}")

            step_outputs = step_module.run(self.config, previous_outputs)

            self.step_outputs[step_name] = step_outputs
            previous_outputs.update(step_outputs)

            print(f"✓ Step {i} complete")

        print("\n" + "="*80)
        print("CORE PIPELINE COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to: {output_path / 'data'}")
        print("  - roi_timeseries_magnitude.set (MNE/EEGLAB compatible)")
        print("  - roi_timeseries_signed.set (MNE/EEGLAB compatible)")

        # Run optional post-processing steps
        if include_spectral:
            print(f"\n{'='*80}")
            print("OPTIONAL: Spectral Analysis")
            print(f"{'='*80}")
            step_outputs = spectral_analysis.run(self.config, previous_outputs)
            self.step_outputs['spectral_analysis'] = step_outputs
            previous_outputs.update(step_outputs)
            print("✓ Spectral analysis complete")

        if include_visualization:
            print(f"\n{'='*80}")
            print("OPTIONAL: Visualization")
            print(f"{'='*80}")
            step_outputs = visualization.run(self.config, previous_outputs)
            self.step_outputs['visualization'] = step_outputs
            previous_outputs.update(step_outputs)
            print("✓ Visualization complete")

        # Generate HTML report (always)
        self._generate_html_report(output_path)

        return self.step_outputs

    def run_spectral_analysis(self) -> Dict[str, Any]:
        """Run spectral analysis on previously computed ROI time series.

        Must be called after run() completes.

        Returns
        -------
        results : dict
            Spectral analysis outputs (band power, etc.)
        """
        if 'roi_extraction' not in self.step_outputs:
            raise RuntimeError("Must run core pipeline first with run()")

        previous_outputs = {}
        for step_outputs in self.step_outputs.values():
            previous_outputs.update(step_outputs)

        print(f"\n{'='*80}")
        print("Spectral Analysis")
        print(f"{'='*80}")

        step_outputs = spectral_analysis.run(self.config, previous_outputs)
        self.step_outputs['spectral_analysis'] = step_outputs

        print("✓ Spectral analysis complete")
        return step_outputs

    def run_visualization(self) -> Dict[str, Any]:
        """Run visualization on previously computed results.

        Must be called after run() completes.

        Returns
        -------
        results : dict
            Visualization outputs
        """
        if 'roi_extraction' not in self.step_outputs:
            raise RuntimeError("Must run core pipeline first with run()")

        previous_outputs = {}
        for step_outputs in self.step_outputs.values():
            previous_outputs.update(step_outputs)

        print(f"\n{'='*80}")
        print("Visualization")
        print(f"{'='*80}")

        step_outputs = visualization.run(self.config, previous_outputs)
        self.step_outputs['visualization'] = step_outputs

        print("✓ Visualization complete")
        return step_outputs

    def _generate_html_report(self, output_path):
        """Generate comprehensive HTML report of pipeline results."""
        from .utils.html_report import generate_pipeline_report

        html_path = output_path / 'pipeline_report.html'
        generate_pipeline_report(
            config=self.config,
            step_outputs=self.step_outputs,
            output_path=html_path
        )
        print(f"\n✓ HTML report saved: {html_path}")
