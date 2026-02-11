"""Step 2: BEM Model Construction.

Build boundary element model (sphere or ellipsoid) for forward modeling.
Implements caching to avoid recomputing BEM models for the same atlas.
"""

import mne
import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime


def get_brain_checksum(brain_file):
    """
    Compute SHA256 checksum of brain volume file for cache validation.

    Parameters
    ----------
    brain_file : Path
        Path to brain volume file

    Returns
    -------
    checksum : str
        SHA256 checksum as hex string
    """
    with open(brain_file, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_bem_cache_path(config):
    """
    Get path to BEM cache directory in the study output folder.

    Parameters
    ----------
    config : Config
        Pipeline configuration

    Returns
    -------
    cache_dir : Path
        Path to BEM cache directory
    """
    # Create cache directory in the study output directory
    output_dir = Path(config['outputs']['dir'])
    cache_dir = output_dir / 'bem_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_bem_cache_files(config):
    """
    Get paths to BEM cache files (model + params).

    Parameters
    ----------
    config : Config
        Pipeline configuration

    Returns
    -------
    model_file : Path
        Path to BEM model pickle file
    params_file : Path
        Path to BEM parameters JSON file
    """
    cache_dir = get_bem_cache_path(config)
    bem_type = config['pipeline']['bem_type']
    n_layers = config['bem'][bem_type]['n_layers']

    cache_name = f"{bem_type}_{n_layers}layer"
    model_file = cache_dir / f"{cache_name}.pkl"
    params_file = cache_dir / f"{cache_name}_params.json"

    return model_file, params_file


def validate_bem_cache(config, params_file):
    """
    Check if cached BEM is valid for current config.

    Validates that:
    - Cache file exists
    - Atlas file hasn't changed (checksum)
    - BEM parameters match current config

    Parameters
    ----------
    config : Config
        Pipeline configuration
    params_file : Path
        Path to cached parameters JSON file

    Returns
    -------
    valid : bool
        True if cache is valid, False otherwise
    """
    if not params_file.exists():
        return False

    try:
        with open(params_file, 'r') as f:
            cached_params = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to load cache metadata: {e}")
        return False

    # Check brain volume hasn't changed
    package_dir = Path(__file__).parent.parent
    brain_file = package_dir / config['inputs']['brain_volume']
    current_checksum = get_brain_checksum(brain_file)

    if cached_params.get('brain_checksum') != current_checksum:
        print("  ⚠️  Brain volume file has changed, BEM cache invalid")
        return False

    # Check BEM type matches
    bem_type = config['pipeline']['bem_type']
    if cached_params.get('bem_type') != bem_type:
        return False

    # Check conductivities match
    if cached_params.get('conductivities') != config['bem'][bem_type]['conductivities']:
        print("  ⚠️  Conductivities changed, BEM cache invalid")
        return False

    # Check number of layers
    if cached_params.get('n_layers') != config['bem'][bem_type]['n_layers']:
        print("  ⚠️  Number of layers changed, BEM cache invalid")
        return False

    print(f"  ✓ Valid BEM cache found (created: {cached_params.get('creation_date', 'unknown')})")
    return True


def save_bem_cache(config, bem_model, bem_params):
    """
    Save BEM model and parameters to cache.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    bem_model : mne.bem.ConductorModel or dict
        BEM model to cache
    bem_params : dict
        BEM parameters
    """
    model_file, params_file = get_bem_cache_files(config)

    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump(bem_model, f)

    # Save metadata
    bem_type = config['pipeline']['bem_type']
    package_dir = Path(__file__).parent.parent
    brain_file = package_dir / config['inputs']['brain_volume']

    cache_metadata = {
        'bem_type': bem_type,
        'n_layers': config['bem'][bem_type]['n_layers'],
        'conductivities': config['bem'][bem_type]['conductivities'],
        'brain_file': brain_file.name,
        'brain_checksum': get_brain_checksum(brain_file),
        'creation_date': datetime.now().isoformat(),
        'package_version': '0.1.0',
        'parameters': bem_params
    }

    with open(params_file, 'w') as f:
        json.dump(cache_metadata, f, indent=2)

    print(f"  ✓ BEM model cached to {model_file.name}")


def run(config, previous_outputs):
    """
    Build BEM model with caching support.

    Workflow:
    1. Check if cached BEM exists and is valid
    2. If valid cache: load and return
    3. If no cache or invalid: create BEM, save to cache, return

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from step 1 (not used for BEM creation)

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'bem': BEM model (mne.bem.ConductorModel or dict)
        - 'bem_type': str - Type of BEM ('sphere' or 'ellipsoid')
        - 'bem_params': dict - BEM parameters (center, radii, etc.)
    """
    bem_type = config['pipeline']['bem_type']
    print(f"BEM Type: {bem_type}")

    # Check if caching is enabled
    use_cache = config['bem'][bem_type].get('use_cache', True)
    force_recreate = config['bem'][bem_type].get('force_recreate', False)

    # Get cache file paths
    model_file, params_file = get_bem_cache_files(config)

    # Try to load from cache
    if use_cache and not force_recreate and model_file.exists() and validate_bem_cache(config, params_file):
        # Load from cache
        print(f"Loading cached BEM model from {model_file.name}...")

        with open(model_file, 'rb') as f:
            bem_model = pickle.load(f)

        with open(params_file, 'r') as f:
            cached_metadata = json.load(f)
            bem_params = cached_metadata['parameters']

        n_layers = cached_metadata.get('n_layers', len(cached_metadata.get('conductivities', [])))
        print(f"✓ Loaded cached {bem_type} BEM ({n_layers} layers)")

    else:
        # Create new BEM
        if force_recreate:
            print(f"Force recreating {bem_type} BEM model (--recreate-bem flag)")
        elif not use_cache:
            print(f"Creating {bem_type} BEM model (caching disabled)")
        else:
            print(f"Creating new {bem_type} BEM model...")
            print("  (This will be cached for future runs)")

        # Call appropriate BEM creation function
        if bem_type == 'sphere':
            from ..bem import sphere
            bem_model, bem_params = sphere.create_bem(config, previous_outputs)
        elif bem_type == 'ellipsoid':
            from ..bem import ellipsoid
            bem_model, bem_params = ellipsoid.create_bem(config, previous_outputs)
        else:
            raise ValueError(f"Unknown BEM type: {bem_type}")

        print(f"✓ Created {bem_type} BEM model")

        # Save to cache (if caching enabled)
        if use_cache:
            save_bem_cache(config, bem_model, bem_params)

    return {
        'bem': bem_model,
        'bem_type': bem_type,
        'bem_params': bem_params
    }
