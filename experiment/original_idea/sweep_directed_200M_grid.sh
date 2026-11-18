#!/usr/bin/env bash
# Converged, same-budget width × depth grid for the directed capacity sweep.
#
# Goal: every (n, width, depth) cell trained to the same budget and past its
# plateau, so the 3D width×depth plots compare converged policies.
#
#   1. 200M grid: n {1,2,4} × [w16,w64 × d1,d3 + w256 d1] × seeds {0,1,2}
#      = 45 fresh runs, 34 evals (~6M eval spacing, same as the other sweeps).
#      Linear (w16 d0) and w256 d3 are NOT rerun: their 300M runs in
#      directed_capacity_sweep_linear300M are scored at the 200M point
#      (constant LR 3e-4, no schedule, so the first 200M of a 300M run is
#      equivalent to a 200M run).
#   2. n=1 linear continuation: n=1 linear is still rising at 300M, so its
#      3 runs are continued from their final params for another 300M
#      (~600M total). Adam state is not saved, so the optimizer restarts.
#
# Waits for tmux sessions large300M and distill so runs never share the GPU.
# Resumable: sweep_directed.sh skips finished runs; the continuation skips
# seeds that already have a finished run.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="$ROOT/experiment/__data__"
LOG="$DATA/grid200M_sweep.out"
SWEEP="$ROOT/experiment/original_idea/sweep_directed.sh"
PY="$ROOT/experiment/mujoco_playground/.venv/bin/python"

for s in large300M distill; do
    while tmux has-session -t "$s" 2>/dev/null; do sleep 60; done
done

grid() {  # $1=N $2=WIDTHS $3=DEPTHS
    N_LIST="$1" WIDTHS="$2" DEPTHS="$3" SEEDS="0 1 2" LINEAR_BASELINE=0 \
    NUM_TIMESTEPS=200000000 NUM_EVALS=34 OUTDIR="$DATA/directed_capacity_sweep_200M" \
        bash "$SWEEP" 2>&1 | tee -a "$LOG"
}

# Whole morphologies first, so a late finish loses n=4 cells, not scattered ones.
for n in 1 2 4; do
    grid "$n" "16 64" "1 3"
    grid "$n" "256" "1"
done

# n=1 linear: continue 300M runs for another 300M.
CONT="$DATA/directed_capacity_n1linear_continue"
mkdir -p "$CONT/_logs"
for s in 0 1 2; do
    if compgen -G "$CONT/insect_n_n1_w16_d0_seed${s}-*/training_history.json" >/dev/null; then
        echo "----- n1 linear seed $s continuation already done, skipping" | tee -a "$LOG"
        continue
    fi
    parent=$(ls -d "$DATA"/directed_capacity_sweep_linear300M/insect_n_n1_w16_d0_seed${s}-*/ | tail -1)
    echo "===== n1 linear seed $s continue from $parent  ($(date '+%H:%M:%S'))" | tee -a "$LOG"
    (cd "$ROOT/experiment" && "$PY" rl/directed_locomotion_ppo_jax.py \
        --morphology insect_n --n-pairs 1 --hidden-width 16 --n-hidden-layers 0 \
        --value-hidden-layer-sizes 256,256,256 --fov-deg 360 --seed "$s" \
        --num-timesteps 300000000 --num-evals 50 --num-envs 2048 \
        --restore-params "${parent}params.pkl" --outdir "$CONT" \
        > "$CONT/_logs/n1_linear_seed${s}.log" 2>&1) \
        || echo "!!!!! n1 linear seed $s continuation FAILED" | tee -a "$LOG"
    grep -E "^\[ *[0-9]+\]" "$CONT/_logs/n1_linear_seed${s}.log" | tail -1 | tee -a "$LOG"
done
echo "grid200M queue complete ($(date '+%H:%M:%S'))" | tee -a "$LOG"
