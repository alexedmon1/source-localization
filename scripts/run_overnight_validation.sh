#!/bin/bash
# Overnight validation batch script
# Runs brain_size, conductivity_ratio, and dipole_size tests
# Parameters: 22 ROI atlas, 25 trials per ROI, 6 parallel jobs

set -e

cd /home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization
source .venv/bin/activate

LOG_DIR="validation/logs/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting overnight validation at $(date)"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run a batch of configs
run_batch() {
    local test_type=$1
    shift
    local configs=("$@")

    echo "=== Running $test_type tests (${#configs[@]} configs) ==="

    # Run in batches of 6
    local batch_size=6
    local total=${#configs[@]}
    local completed=0

    while [ $completed -lt $total ]; do
        local batch=()
        for ((i=0; i<batch_size && completed+i<total; i++)); do
            batch+=("${configs[$completed+$i]}")
        done

        echo "  Batch: ${batch[*]}"

        # Launch batch in parallel
        for config in "${batch[@]}"; do
            local config_name=$(basename "$config" .yaml)
            source-localization validate --test "$test_type" --config "$config_name" --atlas coarse_22roi --trials 25 --quiet > "$LOG_DIR/${config_name}.log" 2>&1 &
        done

        # Wait for batch to complete
        wait

        completed=$((completed + ${#batch[@]}))
        echo "  Completed: $completed / $total"
    done

    echo "  $test_type tests complete!"
    echo ""
}

# Collect all config names
echo "Collecting config names..."

BRAIN_SIZE_CONFIGS=($(ls src/source_localization/validation/config/default_tests/brain_size/*.yaml | grep -v _base | xargs -n1 basename | sed 's/.yaml//'))
CONDUCTIVITY_CONFIGS=($(ls src/source_localization/validation/config/default_tests/conductivity_ratio/*.yaml | grep -v _base | xargs -n1 basename | sed 's/.yaml//'))
DIPOLE_SIZE_CONFIGS=($(ls src/source_localization/validation/config/default_tests/dipole_size/*.yaml | grep -v _base | xargs -n1 basename | sed 's/.yaml//'))

echo "brain_size: ${#BRAIN_SIZE_CONFIGS[@]} configs"
echo "conductivity_ratio: ${#CONDUCTIVITY_CONFIGS[@]} configs"
echo "dipole_size: ${#DIPOLE_SIZE_CONFIGS[@]} configs"
echo "Total: $((${#BRAIN_SIZE_CONFIGS[@]} + ${#CONDUCTIVITY_CONFIGS[@]} + ${#DIPOLE_SIZE_CONFIGS[@]})) configs"
echo ""

# Run tests
run_batch "brain_size" "${BRAIN_SIZE_CONFIGS[@]}"
run_batch "conductivity_ratio" "${CONDUCTIVITY_CONFIGS[@]}"
run_batch "dipole_size" "${DIPOLE_SIZE_CONFIGS[@]}"

echo "=== All validation tests complete! ==="
echo "Finished at $(date)"
echo ""
echo "Results are in:"
echo "  validation/results/brain_size/"
echo "  validation/results/conductivity_ratio/"
echo "  validation/results/dipole_size/"
echo ""
echo "Logs are in: $LOG_DIR"
