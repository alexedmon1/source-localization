#!/bin/bash
# Run key validation configs to compare methods after dSPM fix

cd /home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization
source .venv/bin/activate

# Key configs: sphere volumetric with all methods + best performers
configs=(
    "V01_sphere_vol_dspm"
    "V02_sphere_vol_mne"
    "V03_sphere_vol_sloreta"
    "V21_sphere_roi_sloreta"
)

for config in "${configs[@]}"; do
    echo "Running $config..."
    source-localization validate --test original --config "$config" --atlas coarse_22roi --trials 25 --quiet 2>&1 | grep -E "(ROI Accuracy|Localization Error|Method|Complete)" &
done

wait
echo "All validations complete!"
