#!/usr/bin/env python3
"""Update ROI-based validation configs to use adaptive sources."""

import yaml
from pathlib import Path

base_dir = Path("/home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization/src/source_localization/validation/config/default_tests")

test_dirs = ['brain_size', 'conductivity_ratio', 'dipole_size']

# New ROI source space settings
new_roi_settings = {
    'adaptive_sources': True,
    'max_total_sources': 200,
    'placement_strategy': 'pca'
}

updated_count = 0

for test_dir in test_dirs:
    config_dir = base_dir / test_dir
    if not config_dir.exists():
        continue

    for config_file in config_dir.glob('*.yaml'):
        if config_file.name.startswith('_base'):
            continue

        # Check if it's a ROI config (V15, V21, V24)
        if '_V15_' in config_file.name or '_V21_' in config_file.name or '_V24_' in config_file.name:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Update source_space settings
            if 'source_space' in config and 'roi_based' in config['source_space']:
                old_settings = config['source_space']['roi_based']
                config['source_space']['roi_based'] = new_roi_settings

                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                print(f"Updated: {config_file.name}")
                updated_count += 1

print(f"\nTotal configs updated: {updated_count}")
