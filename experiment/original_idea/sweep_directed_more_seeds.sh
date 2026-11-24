#!/usr/bin/env bash
# Finish the n=6/8 queue, then add seeds to the full 200M directed grid.
#
#   0. Resume sweep_directed_n68.sh (it was killed 2026-11-19 10:58 by a CUDA
#      driver failure at n8 w16 d3 seed0; finished runs are skipped).
#   1. Extra seeds (default 3 4) for every cell: n {1,2,4,6,8} × [linear,
#      w16/w64 × d1/d3, w256 × d1/d3], 200M steps, 34 evals, critic 256x3,
#      all in directed_capacity_sweep_200M. Linear and w256 d3 at n=1,2,4 have
#      seeds 0-2 in the 300M dir (scored at 200M); the new seeds are plain
#      200M runs, which plot_grid_200M.py merges by (n, w, d).
#      Seed-major order: all of seed 3 before seed 4, so stopping early leaves
#      every cell with the same seed count.
#
# Estimated cost: ~14 h per extra seed (n1 ~13 min/run ... n8 ~45 min/run).
#
# A GPU check runs before each batch; if CUDA fails to initialise the queue
# stops instead of burning through every run as a fast failure.
# Resumable. Stop with: tmux kill-session -t seeds

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="$ROOT/experiment/__data__"
LOG="$DATA/more_seeds_sweep.out"
SWEEP="$ROOT/experiment/original_idea/sweep_directed.sh"
PY="$ROOT/experiment/mujoco_playground/.venv/bin/python"
EXTRA_SEEDS="${EXTRA_SEEDS:-3 4}"

gpu_ok() {
    if ! "$PY" -c "import jax; assert jax.devices()[0].platform == 'gpu'" >/dev/null 2>&1; then
        echo "!!!!! GPU unavailable ($(date '+%F %H:%M:%S')) — stopping queue; rerun to resume" | tee -a "$LOG"
        exit 1
    fi
}

echo "##### stage 0: finish n68 queue ($(date '+%F %H:%M:%S'))" | tee -a "$LOG"
gpu_ok
bash "$ROOT/experiment/original_idea/sweep_directed_n68.sh"

grid() {  # $1=N $2=WIDTHS $3=DEPTHS $4=SEED $5=LINEAR
    gpu_ok
    N_LIST="$1" WIDTHS="$2" DEPTHS="$3" SEEDS="$4" LINEAR_BASELINE="$5" \
    NUM_TIMESTEPS=200000000 NUM_EVALS=34 OUTDIR="$DATA/directed_capacity_sweep_200M" \
        bash "$SWEEP" 2>&1 | tee -a "$LOG"
}

for s in $EXTRA_SEEDS; do
    echo "##### stage 1: seed $s, all cells ($(date '+%F %H:%M:%S'))" | tee -a "$LOG"
    for n in 1 2 4 6 8; do
        grid "$n" "16 64" "1 3" "$s" 1   # linear (w16 d0) + w16/w64 × d1/d3
        grid "$n" "256" "1 3" "$s" 0
    done
done
echo "more-seeds queue complete ($(date '+%F %H:%M:%S'))" | tee -a "$LOG"
