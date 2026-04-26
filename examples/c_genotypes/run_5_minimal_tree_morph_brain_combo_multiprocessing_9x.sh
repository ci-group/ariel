#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE="${SCRIPT_DIR}/5_minimal_tree_morph_brain_combo_multiprocessing.py"

for run in $(seq 1 9); do
  echo "=== Run ${run}/9 ==="
  uv run "${EXAMPLE}" --no-visualize "$@"
done
