#!/bin/bash
# Run sensitivity tests in parallel (6 at a time)

cd /home/metalexy/sandbox/AlexProjects/mouse-eeg-source-localization/source_localization
source .venv/bin/activate

PARALLEL=6

run_batch() {
    local test_name=$1
    shift
    local configs=("$@")

    echo ""
    echo "=============================================="
    echo "Running $test_name tests: ${#configs[@]} configs"
    echo "=============================================="

    # Process in batches of $PARALLEL
    for ((i=0; i<${#configs[@]}; i+=PARALLEL)); do
        batch=("${configs[@]:i:PARALLEL}")
        echo ""
        echo "=== Batch $((i/PARALLEL + 1)): ${batch[*]} ==="

        for config in "${batch[@]}"; do
            echo "  Starting $config..."
            source-localization validate --test $test_name --config $config --quiet &
        done

        wait
        echo "=== Batch complete ==="
    done
}

echo "Starting sensitivity tests at $(date)"

# Brain size tests (24 configs)
BRAIN_SIZE=(
    S01_V02_mouse S01_V03_mouse S01_V05_mouse S01_V06_mouse S01_V09_mouse S01_V10_mouse
    S01_V12_mouse S01_V13_mouse S01_V20_mouse S01_V21_mouse S01_V24_mouse
    S02_V02_human S02_V03_human S02_V05_human S02_V06_human S02_V09_human S02_V10_human
    S02_V12_human S02_V13_human S02_V15_human S02_V20_human S02_V21_human S02_V24_human
)
# Note: S01_V15 and S02_V15 may exist - adding S02_V15
run_batch "brain_size" "${BRAIN_SIZE[@]}"

# Conductivity ratio tests (48 configs - C01-C04 x 12 V configs)
COND_RATIO=()
for c in C01 C02 C03 C04; do
    for v in V02 V03 V05 V06 V09 V10 V12 V13 V15 V20 V21 V24; do
        case $c in
            C01) suffix="20to1" ;;
            C02) suffix="40to1" ;;
            C03) suffix="80to1" ;;
            C04) suffix="160to1" ;;
        esac
        COND_RATIO+=("${c}_${v}_${suffix}")
    done
done
run_batch "conductivity_ratio" "${COND_RATIO[@]}"

# Dipole size tests (D01-D11 x various V configs)
DIPOLE=()
# D01-D05: low noise (10, 25, 50, 100, 200 nAm)
for d in D01 D02 D03 D04 D05; do
    for v in V02 V03 V05 V06 V09 V10 V12 V13 V15 V20 V21 V24; do
        case $d in
            D01) amp="10nAm_lownoise" ;;
            D02) amp="25nAm_lownoise" ;;
            D03) amp="50nAm_lownoise" ;;
            D04) amp="100nAm_lownoise" ;;
            D05) amp="200nAm_lownoise" ;;
        esac
        DIPOLE+=("${d}_${v}_${amp}")
    done
done

# D06-D10: high noise
for d in D06 D07 D08 D09 D10; do
    for v in V02 V03 V05 V06 V09 V10 V12 V13 V15 V20 V21 V24; do
        case $d in
            D06) amp="10nAm_highnoise" ;;
            D07) amp="25nAm_highnoise" ;;
            D08) amp="50nAm_highnoise" ;;
            D09) amp="100nAm_highnoise" ;;
            D10) amp="200nAm_highnoise" ;;
        esac
        DIPOLE+=("${d}_${v}_${amp}")
    done
done

# D11: very high noise (subset of V configs)
for v in V03 V15 V21 V24; do
    for amp in 10nAm 25nAm 50nAm 100nAm 200nAm; do
        DIPOLE+=("D11_${v}_${amp}_veryhighnoise")
    done
done

run_batch "dipole_size" "${DIPOLE[@]}"

echo ""
echo "=============================================="
echo "All sensitivity tests complete at $(date)"
echo "=============================================="
