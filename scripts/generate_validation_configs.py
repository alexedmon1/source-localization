#!/usr/bin/env python
"""
Generate validation configs for top 4 performing pipelines across all test types.

Top performers from original validation:
- V24: ellipsoid + roi_based + sLORETA (76.9% ROI accuracy)
- V03: sphere + volumetric + sLORETA (61.1%)
- V15: ellipsoid + roi_based + MNE (57.7%)
- V21: sphere + roi_based + sLORETA (56.6%)
"""

from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent.parent / 'src' / 'source_localization' / 'validation' / 'config' / 'default_tests'

# Top 4 pipeline configurations
PIPELINES = {
    'V24': {
        'bem_type': 'ellipsoid',
        'source_type': 'roi_based',
        'method': 'sLORETA',
        'bem': {
            'ellipsoid': {
                'n_layers': 3,
                'conductivities': [0.33, 0.0042, 0.33],
                'radii_ratios': [0.87, 0.92, 1.0],
                'ellipsoid_method': 'axis_aligned',
                'ellipsoid_margin': 1.23,
                'use_cache': True
            }
        },
        'source_space': {
            'roi_based': {
                'sources_per_roi': 4,
                'jitter_mm': 0.5,
                'include_centroid': True
            }
        }
    },
    'V03': {
        'bem_type': 'sphere',
        'source_type': 'volumetric',
        'method': 'sLORETA',
        'bem': {
            'sphere': {
                'n_layers': 3,
                'conductivities': [0.33, 0.0042, 0.33],
                'radii_ratios': [0.87, 0.92, 1.0],
                'electrode_centered': True,
                'brain_radius_mm': None,
                'use_cache': True,
                'compute_method': 'analytical'
            }
        },
        'source_space': {
            'volumetric': {
                'spacing_mm': 'auto',
                'target_sources_per_channel': 7,
                'max_sources_per_channel': 15,
                'pos_mm': 0.0
            }
        }
    },
    'V15': {
        'bem_type': 'ellipsoid',
        'source_type': 'roi_based',
        'method': 'MNE',
        'bem': {
            'ellipsoid': {
                'n_layers': 3,
                'conductivities': [0.33, 0.0042, 0.33],
                'radii_ratios': [0.87, 0.92, 1.0],
                'ellipsoid_method': 'axis_aligned',
                'ellipsoid_margin': 1.23,
                'use_cache': True
            }
        },
        'source_space': {
            'roi_based': {
                'sources_per_roi': 4,
                'jitter_mm': 0.5,
                'include_centroid': True
            }
        }
    },
    'V21': {
        'bem_type': 'sphere',
        'source_type': 'roi_based',
        'method': 'sLORETA',
        'bem': {
            'sphere': {
                'n_layers': 3,
                'conductivities': [0.33, 0.0042, 0.33],
                'radii_ratios': [0.87, 0.92, 1.0],
                'fit_to_electrodes': False,
                'use_cache': True
            }
        },
        'source_space': {
            'roi_based': {
                'sources_per_roi': 4,
                'jitter_mm': 0.5,
                'include_centroid': True
            }
        }
    }
}

# Dipole size test variations (with fixed noise variance)
# D01-D05: Low noise (1.0 µV²) - baseline, high SNR even at low amplitude
# D06-D10: High noise (25.0 µV²) - challenging, shows SNR effects
DIPOLE_SIZES = {
    # Low noise variants (original) - SNR range: ~10-36 dB
    'D01': {'amplitude_nAm': 10.0, 'noise_uV2': 1.0, 'label': '10nAm_lownoise'},
    'D02': {'amplitude_nAm': 25.0, 'noise_uV2': 1.0, 'label': '25nAm_lownoise'},
    'D03': {'amplitude_nAm': 50.0, 'noise_uV2': 1.0, 'label': '50nAm_lownoise'},
    'D04': {'amplitude_nAm': 100.0, 'noise_uV2': 1.0, 'label': '100nAm_lownoise'},
    'D05': {'amplitude_nAm': 200.0, 'noise_uV2': 1.0, 'label': '200nAm_lownoise'},
    # High noise variants - SNR range: ~-4 to +22 dB
    'D06': {'amplitude_nAm': 10.0, 'noise_uV2': 25.0, 'label': '10nAm_highnoise'},
    'D07': {'amplitude_nAm': 25.0, 'noise_uV2': 25.0, 'label': '25nAm_highnoise'},
    'D08': {'amplitude_nAm': 50.0, 'noise_uV2': 25.0, 'label': '50nAm_highnoise'},
    'D09': {'amplitude_nAm': 100.0, 'noise_uV2': 25.0, 'label': '100nAm_highnoise'},
    'D10': {'amplitude_nAm': 200.0, 'noise_uV2': 25.0, 'label': '200nAm_highnoise'},
}

# Conductivity ratio test variations
CONDUCTIVITY_RATIOS = {
    'C01': {'ratio': '20to1', 'conductivities': [0.33, 0.0165, 0.33], 'label': '20:1 ratio (4x mismatch)'},
    'C02': {'ratio': '40to1', 'conductivities': [0.33, 0.00825, 0.33], 'label': '40:1 ratio (2x mismatch)'},
    'C03': {'ratio': '80to1', 'conductivities': [0.33, 0.0042, 0.33], 'label': '80:1 ratio (no mismatch)'},
    'C04': {'ratio': '160to1', 'conductivities': [0.33, 0.0021, 0.33], 'label': '160:1 ratio (0.5x mismatch)'},
}

# Brain size test variations
BRAIN_SIZES = {
    'S01': {'scale': 1.0, 'label': 'mouse'},
    'S02': {'scale': 11.0, 'label': 'human'},
}


def generate_dipole_size_configs():
    """Generate dipole size test configs with fixed noise variance."""
    output_dir = CONFIG_DIR / 'dipole_size'

    # Remove old configs (except base)
    for f in output_dir.glob('D*.yaml'):
        f.unlink()

    for d_id, d_params in DIPOLE_SIZES.items():
        for p_id, pipeline in PIPELINES.items():
            config_name = f"{d_id}_{p_id}_{d_params['label']}"

            config = {
                '_base': '_base_validation.yaml',
                'pipeline': {
                    'name': f"{d_id}: {d_params['amplitude_nAm']} nAm dipole ({p_id})",
                    'bem_type': pipeline['bem_type'],
                    'source_type': pipeline['source_type']
                },
                'validation': {
                    'dipole': {
                        'amplitude_nAm': d_params['amplitude_nAm'],
                        'noise_mode': 'fixed_variance',
                        'noise_variance_uV2': d_params['noise_uV2']
                    }
                },
                'bem': pipeline['bem'],
                'source_space': pipeline['source_space'],
                'forward': {
                    'meg': False,
                    'eeg': True,
                    'mindist': 0.0
                },
                'inverse': {
                    'method': pipeline['method'],
                    'snr': 3.0,
                    'depth_weighting': 0.8
                },
                'outputs': {
                    'dir': f"validation/results/dipole_size/{config_name}"
                }
            }

            with open(output_dir / f"{config_name}.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(DIPOLE_SIZES) * len(PIPELINES)} dipole_size configs")


def generate_conductivity_configs():
    """Generate conductivity ratio test configs."""
    output_dir = CONFIG_DIR / 'conductivity_ratio'

    # Remove old configs (except base)
    for f in output_dir.glob('C*.yaml'):
        f.unlink()

    ground_truth_conductivities = [0.33, 0.0042, 0.33]  # 80:1 ratio

    for c_id, c_params in CONDUCTIVITY_RATIOS.items():
        for p_id, pipeline in PIPELINES.items():
            config_name = f"{c_id}_{p_id}_{c_params['ratio']}"

            # Deep copy BEM config and update conductivities
            bem_config = {}
            bem_type = pipeline['bem_type']
            bem_config[bem_type] = dict(pipeline['bem'][bem_type])
            bem_config[bem_type]['conductivities'] = c_params['conductivities']

            config = {
                '_base': '_base_validation.yaml',
                'pipeline': {
                    'name': f"{c_id}: {c_params['label']} ({p_id})",
                    'bem_type': pipeline['bem_type'],
                    'source_type': pipeline['source_type']
                },
                'validation': {
                    'forward_model_mismatch': True,
                    'ground_truth_conductivities': ground_truth_conductivities,
                    'test_conductivities': c_params['conductivities']
                },
                'bem': bem_config,
                'source_space': pipeline['source_space'],
                'forward': {
                    'meg': False,
                    'eeg': True,
                    'mindist': 0.0
                },
                'inverse': {
                    'method': pipeline['method'],
                    'snr': 3.0,
                    'depth_weighting': 0.8
                },
                'outputs': {
                    'dir': f"validation/results/conductivity_ratio/{config_name}"
                }
            }

            with open(output_dir / f"{config_name}.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(CONDUCTIVITY_RATIOS) * len(PIPELINES)} conductivity_ratio configs")


def generate_brain_size_configs():
    """Generate brain size test configs."""
    output_dir = CONFIG_DIR / 'brain_size'

    # Remove old configs (except base)
    for f in output_dir.glob('S*.yaml'):
        f.unlink()

    for s_id, s_params in BRAIN_SIZES.items():
        for p_id, pipeline in PIPELINES.items():
            config_name = f"{s_id}_{p_id}_{s_params['label']}"

            config = {
                '_base': '_base_validation.yaml',
                'pipeline': {
                    'name': f"{s_id}: {s_params['label'].capitalize()} scale ({s_params['scale']}x) ({p_id})",
                    'bem_type': pipeline['bem_type'],
                    'source_type': pipeline['source_type']
                },
                'validation': {
                    'scale_factor': s_params['scale']
                },
                'bem': pipeline['bem'],
                'source_space': pipeline['source_space'],
                'forward': {
                    'meg': False,
                    'eeg': True,
                    'mindist': 0.0
                },
                'inverse': {
                    'method': pipeline['method'],
                    'snr': 3.0,
                    'depth_weighting': 0.8
                },
                'outputs': {
                    'dir': f"validation/results/brain_size/{config_name}"
                }
            }

            with open(output_dir / f"{config_name}.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(BRAIN_SIZES) * len(PIPELINES)} brain_size configs")


if __name__ == '__main__':
    generate_dipole_size_configs()
    generate_conductivity_configs()
    generate_brain_size_configs()
    print("\nAll configs generated!")
