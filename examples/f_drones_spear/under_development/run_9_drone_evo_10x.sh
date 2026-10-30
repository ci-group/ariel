#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/9_drone_evo_configurable.py"

RUNS=10
DEVICE="${DEVICE:-cpu}"

echo "Running $SCRIPT $RUNS times on device=$DEVICE"

for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="
    uv run "$SCRIPT" --device "$DEVICE" --seed "$i"
done

echo "All $RUNS runs complete."
