#!/usr/bin/env bash
# Architecture comparison experiment runner.
# Runs N reps × 2 brains × 2 domains, then plots learning curves and renders videos.
#
# Environments
# ------------
# - ARIEL gecko scripts: uv (main ariel venv, Python >=3.12)
#     uv run examples/d_social_learning/<script>.py
#
# - EvoGym scripts: evogym-venv (Python 3.10) — EvoGym requires Python 3.10.
#     Set up with: python3.10 -m venv evogym-venv && evogym-venv/bin/pip install evogym
#     Or set EVOGYM_PYTHON to point to another Python 3.10 with evogym installed.
#
# Usage (from repo root):
#   chmod +x examples/d_social_learning/run_experiment.sh
#   examples/d_social_learning/run_experiment.sh [--gens 50] [--pop 16] [--reps 10]
#
# Outputs (all under __data__/, which is gitignored):
#   __data__/results_gecko/    — gecko .npy histories + best θ
#   __data__/results_evogym/   — evogym .npy histories + best θ
#   __data__/comparison.png    — learning curve plots
#   __data__/vids/             — rendered mp4 videos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENS=50
POP=16
REPS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gens)  GENS="$2";  shift 2 ;;
    --pop)   POP="$2";   shift 2 ;;
    --reps)  REPS="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

EVOGYM_PYTHON="${EVOGYM_PYTHON:-evogym-venv/bin/python}"
if [[ ! -x "$EVOGYM_PYTHON" ]]; then
  echo "ERROR: EvoGym Python not found at $EVOGYM_PYTHON"
  echo "Set up the evogym-venv first (see README.md) or set EVOGYM_PYTHON."
  exit 1
fi

echo "=== ARIEL Gecko experiment (gens=$GENS pop=$POP reps=$REPS) ==="
uv run "$SCRIPT_DIR/experiment_gecko.py" \
  --gens "$GENS" --pop "$POP" --reps "$REPS" \
  --out-dir __data__/results_gecko

echo ""
echo "=== EvoGym experiment (gens=$GENS pop=$POP reps=$REPS) ==="
"$EVOGYM_PYTHON" "$SCRIPT_DIR/experiment_evogym.py" \
  --gens "$GENS" --pop "$POP" --reps "$REPS" \
  --out-dir __data__/results_evogym

echo ""
echo "=== Plotting learning curves ==="
uv run "$SCRIPT_DIR/plot_results.py" \
  --gecko-dir __data__/results_gecko \
  --evogym-dir __data__/results_evogym \
  --out __data__/comparison.png

echo ""
echo "=== Rendering gecko videos ==="
uv run "$SCRIPT_DIR/make_gecko_video_compare.py" \
  --results-dir __data__/results_gecko \
  --out-dir __data__/vids

echo ""
echo "=== Rendering EvoGym videos ==="
"$EVOGYM_PYTHON" "$SCRIPT_DIR/make_evogym_video_compare.py" \
  --results-dir __data__/results_evogym \
  --out-dir __data__/vids

echo ""
echo "=== All done ==="
echo "  __data__/comparison.png"
echo "  __data__/vids/gecko_distributed.mp4 / gecko_standard.mp4"
echo "  __data__/vids/evogym_distributed.mp4 / evogym_standard.mp4"
