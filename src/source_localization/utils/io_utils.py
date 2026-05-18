"""I/O utilities for saving and loading pipeline data."""

import pickle
import numpy as np
from pathlib import Path


def save_pickle(data, filepath):
    """Save data to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_numpy(data, filepath):
    """Save numpy array to .npy file."""
    np.save(filepath, data)


def load_numpy(filepath):
    """Load numpy array from .npy file."""
    return np.load(filepath)


def get_data_dir(config):
    """Get data directory path from config."""
    return Path(config['outputs']['dir']) / 'data'


def get_figures_dir(config):
    """Get figures directory path from config."""
    return Path(config['outputs']['dir']) / 'figures'


def get_output_variants(config):
    """Normalize ``outputs.output_variants`` to a list of variant names.

    Accepts the string forms ``"signed"``, ``"magnitude"``, ``"both"`` or
    an explicit list. Returns a list containing one or both of
    ``"signed"`` and ``"magnitude"``.
    """
    raw = config.get('outputs', {}).get('output_variants', 'both')
    if isinstance(raw, str):
        if raw == 'both':
            return ['signed', 'magnitude']
        if raw in ('signed', 'magnitude'):
            return [raw]
        raise ValueError(
            f"outputs.output_variants={raw!r} not recognized. "
            f"Use 'signed', 'magnitude', 'both', or a list."
        )
    variants = list(raw)
    bad = [v for v in variants if v not in ('signed', 'magnitude')]
    if bad:
        raise ValueError(
            f"outputs.output_variants contains unknown variant(s) {bad}; "
            f"allowed values: 'signed', 'magnitude'."
        )
    if not variants:
        raise ValueError("outputs.output_variants must not be empty.")
    return variants
