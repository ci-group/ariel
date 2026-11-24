#!/usr/bin/env bash
# Extend the 200M directed capacity grid to n=6 and n=8.
#
# Same settings as sweep_directed_200M_grid.sh (200M steps, 34 evals, seeds
# 0-2, critic 256x3), written into the same directory so plot_grid_200M.py and
# plot_efficiency.py pick the new cells up automatically.
#
# Stage 1 (~8.5 h): linear + w256 d3 at n=6, then n=8. This is what tests
#   whether the linear-vs-large gap keeps shrinking after n=2.
# Stage 2 (~21 h): the remaining 5 architectures, most informative first
#   (w64 d3, then w256 d1, w64 d1, w16 d3, w16 d1), n=6 before n=8 within each.
#
# Measured cost (2026-11-18 smoke test): n=8 w256 d3 ~89 s per 6.9M steps
# (~50 min per 200M run), 2.9 GB GPU; contacts peak ~20/env vs budget 320.
# n=6 not measured; estimated ~35 min per run.
#
# Resumable: finished runs are skipped. Stop with: tmux kill-session -t n68

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="$ROOT/experiment/__data__"
LOG="$DATA/n68_sweep.out"
SWEEP="$ROOT/experiment/original_idea/sweep_directed.sh"

grid() {  # $1=N $2=WIDTHS $3=DEPTHS
    N_LIST="$1" WIDTHS="$2" DEPTHS="$3" SEEDS="0 1 2" LINEAR_BASELINE=0 \
    NUM_TIMESTEPS=200000000 NUM_EVALS=34 OUTDIR="$DATA/directed_capacity_sweep_200M" \
        bash "$SWEEP" 2>&1 | tee -a "$LOG"
}

echo "##### stage 1: linear + w256 d3 ($(date '+%F %H:%M:%S'))" | tee -a "$LOG"
for n in 6 8; do
    grid "$n" "16" "0"     # linear (width ignored at depth 0)
    grid "$n" "256" "3"
done

echo "##### stage 2: remaining architectures ($(date '+%F %H:%M:%S'))" | tee -a "$LOG"
for arch in "64 3" "256 1" "64 1" "16 3" "16 1"; do
    set -- $arch
    for n in 6 8; do
        grid "$n" "$1" "$2"
    done
done
echo "n68 queue complete ($(date '+%F %H:%M:%S'))" | tee -a "$LOG"
