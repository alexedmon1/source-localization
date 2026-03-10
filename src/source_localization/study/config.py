"""
Study Configuration Module

Defines the configuration structure for multi-subject EEG studies with source localization.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class SubjectInfo:
    """Information about a single subject in the study."""

    subject_id: str
    eeg_file: Path
    group: Optional[str] = None
    session: Optional[str] = None
    recording: Optional[int] = None
    date: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure eeg_file is a Path object."""
        if isinstance(self.eeg_file, str):
            self.eeg_file = Path(self.eeg_file)

    @property
    def output_id(self) -> str:
        """Generate output folder name for this subject."""
        parts = [f"sub-{self.subject_id}"]
        if self.session:
            parts.append(f"ses-{self.session}")
        if self.recording:
            parts.append(f"rec-{self.recording}")
        return "_".join(parts)


@dataclass
class StudyConfig:
    """
    Configuration for a multi-subject source localization study.

    Attributes:
        name: Study name/identifier
        root_dir: Root directory of the study
        pipeline_preset: Source localization preset to use
        pipeline_overrides: Optional parameter overrides
        subjects: List of subjects to process
        discovery_config: Optional auto-discovery configuration
        analysis_config: Configuration for ROI analysis
        output_config: Configuration for output structure
    """

    name: str
    root_dir: Path
    pipeline_preset: str = "roi_based_ellipsoid"
    pipeline_atlas: Optional[str] = None
    pipeline_overrides: Dict[str, Any] = field(default_factory=dict)
    subjects: List[SubjectInfo] = field(default_factory=list)
    discovery_config: Optional[Dict[str, Any]] = None

    # Analysis configuration
    analysis_config: Dict[str, Any] = field(default_factory=lambda: {
        'frequency_bands': {
            'delta': (1, 4),
            'theta': (4, 10),
            'alpha': (10, 13),
            'beta': (13, 30),
            'low_gamma': (30, 55),
            'high_gamma': (65, 100),
        },
        'connectivity_methods': ['correlation', 'coherence'],
        'depth_weighting': True,
        'max_depth_mm': 3.0,
    })

    # Output configuration
    output_config: Dict[str, Any] = field(default_factory=lambda: {
        'save_source_timeseries': True,
        'save_roi_timeseries': True,
        'save_band_power': True,
        'save_connectivity': True,
        'save_figures': True,
        'figure_formats': ['png'],
    })

    def __post_init__(self):
        """Ensure root_dir is a Path object."""
        if isinstance(self.root_dir, str):
            self.root_dir = Path(self.root_dir)

    @property
    def sourcedata_dir(self) -> Path:
        """Directory containing source EEG files."""
        return self.root_dir / "sourcedata"

    @property
    def derivatives_dir(self) -> Path:
        """Directory for processed outputs."""
        return self.root_dir / "derivatives"

    @property
    def group_dir(self) -> Path:
        """Directory for group-level results."""
        return self.derivatives_dir / "group"

    def get_subject_output_dir(self, subject: Union[str, SubjectInfo]) -> Path:
        """Get output directory for a specific subject."""
        if isinstance(subject, str):
            # Find subject by ID
            for s in self.subjects:
                if s.subject_id == subject:
                    subject = s
                    break
            else:
                # Not found, create a simple output path
                return self.derivatives_dir / f"sub-{subject}"
        return self.derivatives_dir / subject.output_id

    def get_subject_by_id(self, subject_id: str) -> Optional[SubjectInfo]:
        """Find a subject by their ID."""
        for subject in self.subjects:
            if subject.subject_id == subject_id:
                return subject
        return None

    def get_subjects_by_group(self, group: str) -> List[SubjectInfo]:
        """Get all subjects belonging to a specific group."""
        return [s for s in self.subjects if s.group == group]

    @property
    def groups(self) -> List[str]:
        """Get unique group names in the study."""
        groups = set(s.group for s in self.subjects if s.group)
        return sorted(groups)

    def to_dict(self, include_resolved_subjects: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Args:
            include_resolved_subjects: If True, include resolved subjects list
                even when discovery_config is set. If False (default), only
                include discovery config when available.
        """
        # Convert analysis config - tuples to lists for YAML compatibility
        analysis_config_safe = {}
        for key, val in self.analysis_config.items():
            if key == 'frequency_bands':
                # Convert tuples to lists
                analysis_config_safe[key] = {
                    band: list(freq_range) if isinstance(freq_range, tuple) else freq_range
                    for band, freq_range in val.items()
                }
            else:
                analysis_config_safe[key] = val

        pipeline_dict = {
            'preset': self.pipeline_preset,
            'overrides': self.pipeline_overrides,
        }
        if self.pipeline_atlas:
            pipeline_dict['atlas'] = self.pipeline_atlas

        result = {
            'name': self.name,
            'pipeline': pipeline_dict,
            'analysis': analysis_config_safe,
            'output': self.output_config,
        }

        # Include discovery config if set
        if self.discovery_config:
            result['discovery'] = self.discovery_config
            # Only include subjects if explicitly requested
            if include_resolved_subjects:
                result['subjects'] = [
                    {
                        'id': s.subject_id,
                        'eeg_file': str(s.eeg_file),
                        'group': s.group,
                        'session': s.session,
                        'recording': s.recording,
                        'date': s.date,
                        'notes': s.notes,
                        'metadata': s.metadata,
                    }
                    for s in self.subjects
                ]
        else:
            # No discovery config, include explicit subjects list
            result['subjects'] = [
                {
                    'id': s.subject_id,
                    'eeg_file': str(s.eeg_file),
                    'group': s.group,
                    'session': s.session,
                    'recording': s.recording,
                    'date': s.date,
                    'notes': s.notes,
                    'metadata': s.metadata,
                }
                for s in self.subjects
            ]

        return result

    def save_yaml(self, path: Optional[Path] = None) -> Path:
        """Save configuration to YAML file."""
        if path is None:
            path = self.root_dir / "study_config.yaml"
        path = Path(path)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any], root_dir: Path) -> 'StudyConfig':
        """
        Create configuration from dictionary.

        Supports two modes:
        1. Explicit subjects list: subjects defined inline
        2. Discovery mode: auto-discover subjects from folder structure

        Discovery config example:
            discovery:
              participants_csv: "participants.csv"  # Optional: CSV with metadata
              eeg_pattern: "**/*.set"               # Glob pattern for EEG files
              group_from_folder: true               # Use parent folder as group
        """
        subjects = []

        # Check for discovery mode
        discovery_config = data.get('discovery')
        if discovery_config:
            # Auto-discover subjects from folder
            eeg_pattern = discovery_config.get('eeg_pattern', '**/*.set')
            participants_csv = discovery_config.get('participants_csv')
            group_from_folder = discovery_config.get('group_from_folder', True)

            # Find all EEG files
            eeg_files = sorted(root_dir.glob(eeg_pattern))

            # Load participants metadata if available
            participants_meta = {}
            recording_to_subject = {}
            if participants_csv:
                participants_path = root_dir / participants_csv if not Path(participants_csv).is_absolute() else Path(participants_csv)
                if participants_path.exists():
                    participants_meta, recording_to_subject = cls._load_participants_csv(participants_path)

            # Create subject entries from discovered files
            for eeg_file in eeg_files:
                filename = eeg_file.stem

                # Try to match file to participants CSV by recording filename
                subject_id = None
                meta = {}

                # Check recording_to_subject mapping (match by core filename)
                for recording_name, sid in recording_to_subject.items():
                    # The actual file may have 'D' prefix and '_data_postcomp' suffix
                    filename_core = filename[1:] if filename.startswith('D') else filename
                    filename_core = filename_core.replace('_data_postcomp', '')

                    recording_core = recording_name[1:] if recording_name.startswith('D') else recording_name

                    if recording_core in filename_core or filename_core in recording_core:
                        subject_id = sid
                        meta = participants_meta.get(sid, {})
                        break

                # Fall back to extracting from filename
                if subject_id is None:
                    subject_id = cls._extract_subject_id(filename)
                    meta = participants_meta.get(subject_id, {})

                # Determine group
                group = None
                if group_from_folder:
                    group = eeg_file.parent.name
                    if group == root_dir.name:
                        group = None

                subjects.append(SubjectInfo(
                    subject_id=subject_id,
                    eeg_file=eeg_file,
                    group=meta.get('group', group),
                    session=meta.get('session'),
                    recording=meta.get('recording'),
                    date=meta.get('date'),
                    notes=meta.get('notes'),
                    metadata={k: v for k, v in meta.items()
                              if k not in ['group', 'session', 'recording', 'date', 'notes']},
                ))

        # Also load explicitly listed subjects (can be used alongside or instead of discovery)
        for s in data.get('subjects', []):
            eeg_file = Path(s['eeg_file'])
            if not eeg_file.is_absolute():
                eeg_file = root_dir / eeg_file

            subjects.append(SubjectInfo(
                subject_id=str(s['id']),
                eeg_file=eeg_file,
                group=s.get('group'),
                session=s.get('session'),
                recording=s.get('recording'),
                date=s.get('date'),
                notes=s.get('notes'),
                metadata=s.get('metadata', {}),
            ))

        pipeline_config = data.get('pipeline', {})

        # Convert frequency_bands from lists to tuples
        analysis_config = data.get('analysis', {})
        if 'frequency_bands' in analysis_config:
            analysis_config['frequency_bands'] = {
                band: tuple(freq_range) if isinstance(freq_range, list) else freq_range
                for band, freq_range in analysis_config['frequency_bands'].items()
            }

        # Merge with defaults
        default_analysis = {
            'frequency_bands': {
                'delta': (1, 4),
                'theta': (4, 10),
                'alpha': (10, 13),
                'beta': (13, 30),
                'low_gamma': (30, 55),
                'high_gamma': (65, 100),
            },
            'connectivity_methods': ['correlation', 'coherence'],
            'depth_weighting': True,
            'max_depth_mm': 3.0,
        }
        for key, val in default_analysis.items():
            if key not in analysis_config:
                analysis_config[key] = val

        return cls(
            name=data.get('name', 'unnamed_study'),
            root_dir=root_dir,
            pipeline_preset=pipeline_config.get('preset', 'roi_based_ellipsoid'),
            pipeline_atlas=pipeline_config.get('atlas'),
            pipeline_overrides=pipeline_config.get('overrides', {}),
            subjects=subjects,
            discovery_config=discovery_config,
            analysis_config=analysis_config,
            output_config=data.get('output', {}),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'StudyConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, root_dir=path.parent)

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        name: Optional[str] = None,
        participants_file: Optional[str] = None,
        eeg_pattern: str = "**/*.set",
        group_from_folder: bool = True,
    ) -> 'StudyConfig':
        """
        Create study configuration by scanning a folder for EEG files.

        Args:
            folder: Root folder containing EEG data
            name: Study name (defaults to folder name)
            participants_file: Path to CSV with subject metadata
            eeg_pattern: Glob pattern to find EEG files
            group_from_folder: If True, use parent folder name as group

        Returns:
            StudyConfig with discovered subjects
        """
        folder = Path(folder)
        name = name or folder.name

        # Find all EEG files
        eeg_files = sorted(folder.glob(eeg_pattern))

        # Load participants metadata if available
        participants_meta = {}
        recording_to_subject = {}  # Map recording filename to subject ID
        if participants_file:
            participants_path = folder / participants_file if not Path(participants_file).is_absolute() else Path(participants_file)
            if participants_path.exists():
                participants_meta, recording_to_subject = cls._load_participants_csv(participants_path)

        # Create subject entries
        subjects = []
        for eeg_file in eeg_files:
            filename = eeg_file.stem

            # Try to match file to participants CSV by recording filename
            subject_id = None
            meta = {}

            # Check recording_to_subject mapping (match by core filename)
            for recording_name, sid in recording_to_subject.items():
                # The actual file may have 'D' prefix and '_data_postcomp' suffix
                # Remove 'D' prefix from filename for matching
                filename_core = filename[1:] if filename.startswith('D') else filename
                filename_core = filename_core.replace('_data_postcomp', '')

                # Also try without 'D' prefix on the recording name
                recording_core = recording_name[1:] if recording_name.startswith('D') else recording_name

                if recording_core in filename_core or filename_core in recording_core:
                    subject_id = sid
                    meta = participants_meta.get(sid, {})
                    break

            # Fall back to extracting from filename
            if subject_id is None:
                subject_id = cls._extract_subject_id(filename)
                meta = participants_meta.get(subject_id, {})

            # Determine group
            group = None
            if group_from_folder:
                # Use immediate parent folder as group
                group = eeg_file.parent.name
                if group == folder.name:
                    group = None

            subjects.append(SubjectInfo(
                subject_id=subject_id,
                eeg_file=eeg_file,
                group=meta.get('group', group),
                session=meta.get('session'),
                recording=meta.get('recording'),
                date=meta.get('date'),
                notes=meta.get('notes'),
                metadata={k: v for k, v in meta.items()
                          if k not in ['group', 'session', 'recording', 'date', 'notes']},
            ))

        return cls(
            name=name,
            root_dir=folder,
            subjects=subjects,
        )

    @staticmethod
    def _extract_subject_id(filename: str) -> str:
        """
        Extract subject ID from filename.

        Handles patterns like:
        - DFX9_947_0__uid0529-19-03-47_data_postcomp
        - Dsbpro_0__uid1219-16-17-24_data_postcomp
        - sub-001_eeg
        """
        import re

        # Try pattern: DFX9_XXX or DFS9_XXX (3-digit ID after prefix)
        match = re.search(r'[DF][SX]\d+_(\d{3})', filename)
        if match:
            return match.group(1)

        # Try pattern: sub-XXX
        match = re.search(r'sub-(\w+)', filename)
        if match:
            return match.group(1)

        # Try to find any 3-digit number
        match = re.search(r'(\d{3})', filename)
        if match:
            return match.group(1)

        # Fall back to using the filename stem (before first underscore)
        return filename.split('_')[0]

    @staticmethod
    def _load_participants_csv(path: Path) -> tuple:
        """
        Load participants metadata from CSV file.

        Returns:
            Tuple of (participants_dict, recording_to_subject_dict)
        """
        participants = {}
        recording_to_subject = {}

        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get subject ID (try various column names)
                subject_id = (
                    row.get('subject_id') or
                    row.get('mouseid') or
                    row.get('id') or
                    row.get('participant_id')
                )
                if subject_id:
                    subject_id = str(subject_id)
                    participants[subject_id] = {
                        'group': row.get('group'),
                        'session': row.get('session'),
                        'recording': row.get('recording_number') or row.get('recording'),
                        'date': row.get('date'),
                        'notes': row.get('notes'),
                        **{k: v for k, v in row.items()
                           if k not in ['subject_id', 'mouseid', 'id', 'participant_id',
                                        'group', 'session', 'recording_number', 'recording',
                                        'date', 'notes']}
                    }

                    # Also map recording filename to subject ID
                    recording_file = row.get('recordingfile')
                    if recording_file:
                        recording_to_subject[recording_file] = subject_id

        return participants, recording_to_subject

    def validate(self) -> List[str]:
        """
        Validate the study configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check root directory exists
        if not self.root_dir.exists():
            errors.append(f"Root directory does not exist: {self.root_dir}")

        # Check all EEG files exist
        for subject in self.subjects:
            if not subject.eeg_file.exists():
                errors.append(f"EEG file not found for subject {subject.subject_id}: {subject.eeg_file}")

        # Check for duplicate subject IDs
        ids = [s.subject_id for s in self.subjects]
        duplicates = [id for id in set(ids) if ids.count(id) > 1]
        if duplicates:
            errors.append(f"Duplicate subject IDs: {duplicates}")

        # Validate pipeline preset exists
        from ..config import Config
        try:
            Config.from_preset(self.pipeline_preset)
        except Exception as e:
            errors.append(f"Invalid pipeline preset '{self.pipeline_preset}': {e}")

        return errors

    def summary(self) -> str:
        """Generate a summary of the study configuration."""
        lines = [
            f"Study: {self.name}",
            f"Root directory: {self.root_dir}",
            f"Pipeline preset: {self.pipeline_preset}",
            f"Atlas: {self.pipeline_atlas or 'antwerp (default)'}",
            f"Number of subjects: {len(self.subjects)}",
        ]

        if self.groups:
            lines.append(f"Groups: {', '.join(self.groups)}")
            for group in self.groups:
                n = len(self.get_subjects_by_group(group))
                lines.append(f"  - {group}: {n} subjects")

        return "\n".join(lines)
