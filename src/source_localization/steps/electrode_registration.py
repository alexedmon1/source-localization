"""Step 1: Electrode Registration.

Register electrodes to MRI space using atlas-based coordinate transformation.
Uses the validated electrode_registration module from adv_test.
"""

import mne
import numpy as np
from pathlib import Path


def run(config, previous_outputs):
    """
    Register electrodes to MRI space.

    Parameters
    ----------
    config : Config
        Pipeline configuration
    previous_outputs : dict
        Outputs from previous steps (empty for step 1)

    Returns
    -------
    outputs : dict
        Dictionary containing:
        - 'info': mne.Info - MNE info object with electrode positions
        - 'electrodes_mri': ndarray, shape (n_channels, 3) - Electrode positions in MRI coordinates (mm)
        - 'ch_names': list - Channel names
        - 'electrode_results': dict - Validation results from electrode registration
    """
    print("Electrode Registration")

    # Import electrode registration module
    from ..utils.electrode_registration import load_electrodes_from_p100

    # Get paths
    package_dir = Path(__file__).parent.parent
    electrodes_csv = package_dir / config['inputs']['electrodes_csv']
    atlas_nii = package_dir / config['inputs']['brain_volume']

    # Get electrode registration parameters from config
    projection_method = config['electrode'].get('projection_method', 'intensity')
    skull_offset_mm = config['electrode'].get('skull_offset_mm', 0.0)
    bregma_vox = config['electrode'].get('bregma_vox', None)
    lambda_vox = config['electrode'].get('lambda_vox', None)

    # Convert bregma/lambda from list to tuple if provided
    if bregma_vox is not None:
        bregma_vox = tuple(bregma_vox)
    if lambda_vox is not None:
        lambda_vox = tuple(lambda_vox)

    print(f"  Electrodes CSV: {electrodes_csv.name}")
    print(f"  Atlas NIfTI: {atlas_nii.name}")
    print(f"  Projection method: {projection_method}")
    print(f"  Skull offset: {skull_offset_mm} mm")

    # Load electrodes using validated P100 method
    info, electrode_results = load_electrodes_from_p100(
        electrodes_csv=electrodes_csv,
        atlas_nii=atlas_nii,
        projection_method=projection_method,
        skull_offset_mm=skull_offset_mm,
        bregma_vox=bregma_vox,
        lambda_vox=lambda_vox,
        create_visualization=config['electrode'].get('create_visualization', False),
        output_dir=Path(config['outputs']['dir']) / 'electrode_registration',
        sfreq=1000.0  # Default sfreq, will be updated from EEG data in later steps
    )

    # Extract electrode coordinates in mm
    n_electrodes = len(info['ch_names'])
    electrodes_mri = np.array([info['chs'][i]['loc'][:3] for i in range(n_electrodes)]) * 1000  # to mm

    print(f"  ✓ Registered {n_electrodes} electrodes")
    print(f"    Electrode range X: [{electrodes_mri[:, 0].min():.2f}, {electrodes_mri[:, 0].max():.2f}] mm")
    print(f"    Electrode range Y: [{electrodes_mri[:, 1].min():.2f}, {electrodes_mri[:, 1].max():.2f}] mm")
    print(f"    Electrode range Z: [{electrodes_mri[:, 2].min():.2f}, {electrodes_mri[:, 2].max():.2f}] mm")

    # Save intermediate data and create visualizations
    if config['outputs'].get('save_intermediate', True):
        from ..utils.io_utils import save_pickle, get_data_dir, get_figures_dir
        from ..utils.step_visualizations import visualize_step1_electrodes

        data_dir = get_data_dir(config)
        figures_dir = get_figures_dir(config)

        # Save info object
        save_pickle(info, data_dir / 'step1_info.pkl')
        print(f"    Saved: {data_dir / 'step1_info.pkl'}")

        # Create electrode visualization
        fig = visualize_step1_electrodes(info, figures_dir / 'step1_electrodes.png')
        print(f"    Saved: {figures_dir / 'step1_electrodes.png'}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    return {
        'info': info,
        'electrodes_mri': electrodes_mri,
        'ch_names': info['ch_names'],
        'electrode_results': electrode_results
    }
