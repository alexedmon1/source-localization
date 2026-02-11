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
