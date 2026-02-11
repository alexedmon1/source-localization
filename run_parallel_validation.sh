#!/bin/bash
# Run original validation tests in parallel (6 at a time)

cd /home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization
source .venv/bin/activate

# Remaining configs (V04-V26, skipping V01-V03 which are done)
CONFIGS=(
    "V04 V05 V06 V07 V08 V09"
    "V10 V11 V12 V13 V14 V15"
    "V16 V17 V18 V19 V20 V21"
    "V22 V23 V24 V25 V26"
)

echo "Starting parallel validation at $(date)"
echo "Running 6 configs at a time..."

for batch in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Starting batch: $batch ==="

    # Start each config in background
    for config in $batch; do
        echo "  Starting $config..."
        source-localization validate --test original --config $config --quiet &
    done

    # Wait for all background jobs to complete
    wait
    echo "=== Batch complete ==="
done

echo ""
echo "All validations complete at $(date)"
