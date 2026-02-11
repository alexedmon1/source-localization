#!/bin/bash
cd /home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization
source .venv/bin/activate

# Run validations in parallel
for config in "$@"; do
    echo "Starting $config..."
    source-localization validate --test original --config "$config" --atlas coarse_22roi --trials 25 --quiet > "/tmp/${config}.log" 2>&1 &
done

echo "Waiting for all jobs to complete..."
wait
echo "All jobs complete"

# Show results summary
for config in "$@"; do
    if [ -f "/tmp/${config}.log" ]; then
        echo "=== $config ==="
        tail -5 "/tmp/${config}.log"
    fi
done
