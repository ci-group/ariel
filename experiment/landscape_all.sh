#!/usr/bin/env bash
# Render the fitness landscape (PCA basis, no filter normalization) for every
# training run in __data__/undirected_locomotion_ppo_jax/.
#
# Runs sequentially so the GPU isn't fought over. Each landscape takes ~1 min
# (eval ~3 s + plotting + optional animation).
#
# Usage (from anywhere):
#   bash experiment/landscape_all.sh                  # all PPO runs, --animate on
#   DATA_SUBDIR=undirected_locomotion_ppo_jax \
#       bash experiment/landscape_all.sh              # explicit subdir
#   PATTERN='insect_n_n3_*' bash experiment/landscape_all.sh    # only n_pairs=3
#   EXTRA_ARGS='--grid 15' bash experiment/landscape_all.sh     # smaller/faster grid
#   ANIMATE=0 bash experiment/landscape_all.sh                  # skip the gif

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENT_DIR"

PY="$EXPERIMENT_DIR/mujoco_playground/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "error: expected mujoco_playground venv interpreter at $PY" >&2
    exit 1
fi

LANDSCAPE_SCRIPT="plotting/fitness_landscape_ppo_pca.py"
if [[ ! -f "$LANDSCAPE_SCRIPT" ]]; then
    echo "error: landscape script not found at $LANDSCAPE_SCRIPT" >&2
    exit 1
fi

DATA_SUBDIR="${DATA_SUBDIR:-undirected_locomotion_ppo_jax}"
DATA_ROOT="$EXPERIMENT_DIR/__data__/$DATA_SUBDIR"
PATTERN="${PATTERN:-*}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ANIMATE="${ANIMATE:-1}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "error: no run dir at $DATA_ROOT" >&2
    exit 1
fi

ANIMATE_FLAG=""
if [[ "$ANIMATE" == "1" ]]; then
    ANIMATE_FLAG="--animate"
fi

# Shell globs don't expand to a sorted list when nullglob is off; collect
# matches, sort, filter runs that actually have a snapshot file.
shopt -s nullglob
candidates=("$DATA_ROOT"/$PATTERN/)
shopt -u nullglob

runs=()
for d in "${candidates[@]}"; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/params_snapshots.pkl" ]]; then
        runs+=("$d")
    else
        echo "skip: $(basename "$d")  (no params_snapshots.pkl)"
    fi
done
IFS=$'\n' runs=($(printf '%s\n' "${runs[@]}" | sort))
unset IFS

total=${#runs[@]}
if (( total == 0 )); then
    echo "no runs with params_snapshots.pkl under $DATA_ROOT (pattern=$PATTERN)" >&2
    exit 1
fi

echo "landscape sweep: $total run(s) in $DATA_ROOT"
echo "                 --no-filter-normalize $ANIMATE_FLAG $EXTRA_ARGS"
echo

failures=()
i=0
for run_dir in "${runs[@]}"; do
    i=$((i + 1))
    name="$(basename "${run_dir%/}")"
    echo "===== [$i/$total] $name ====="
    if "$PY" "$LANDSCAPE_SCRIPT" \
        --run-dir "$run_dir" \
        --no-filter-normalize \
        $ANIMATE_FLAG \
        $EXTRA_ARGS; then
        echo
    else
        echo "FAILED: $name  (continuing)" >&2
        failures+=("$name")
        echo
    fi
done

echo "landscape sweep complete."
if (( ${#failures[@]} > 0 )); then
    echo "failed runs (${#failures[@]}):"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
