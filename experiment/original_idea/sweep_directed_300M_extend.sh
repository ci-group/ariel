#!/usr/bin/env bash
# Extension of the 300M linear-vs-largest comparison (directed capacity sweep).
#
# Only the two architectures the claim rests on: linear (w16 d0) and the
# largest MLP (w256 d3), both at 300M steps, 50 evals (same ~6M eval spacing
# as the 100M sweep). Ordered by value, so a late overrun only loses the tail:
#   1. n=3, seeds 0-2         -> fills in the n axis between 2 and 4
#   2. n=6, seeds 0-2         -> does the trend continue past n=4? (untested size)
#   3. n=1,2,4, seeds 3-4     -> 5 seeds per cell in the main comparison
#
# Waits for the tmux sessions `linear300M` and `large300M` to finish first so
# runs never share the GPU. Resumable (sweep_directed.sh skips finished runs).

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/experiment/__data__/directed_capacity_sweep_linear300M"
LOG="$ROOT/experiment/__data__/linear300M_sweep.out"
SWEEP="$ROOT/experiment/original_idea/sweep_directed.sh"

for s in linear300M large300M; do
    while tmux has-session -t "$s" 2>/dev/null; do sleep 60; done
done

run() {  # $1=N_LIST $2=WIDTHS $3=DEPTHS $4=SEEDS
    N_LIST="$1" WIDTHS="$2" DEPTHS="$3" SEEDS="$4" LINEAR_BASELINE=0 \
    NUM_TIMESTEPS=300000000 NUM_EVALS=50 OUTDIR="$OUT" \
        bash "$SWEEP" 2>&1 | tee -a "$LOG"
}

for n in 3 6; do
    run "$n" 16 0 "0 1 2"
    run "$n" 256 3 "0 1 2"
done
run "1 2 4" 16 0 "3 4"
run "1 2 4" 256 3 "3 4"
