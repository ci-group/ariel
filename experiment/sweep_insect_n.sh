#!/usr/bin/env bash
# Sequential training sweep over the insect_n morphology.
#
# One run per n_pairs value; a single seed each. Each run takes ~4 min on a
# 12 GB laptop GPU at --num-envs 3072 / --num-timesteps 50M.
#
# Runs land in __data__/undirected_locomotion_ppo_jax/insect_n_n{N}_seed{S}-<timestamp>/
# so they stay unambiguous when the fitness-landscape pipeline iterates over them.
#
# Usage (from anywhere):
#   bash experiment/sweep_insect_n.sh                 # defaults: n=1..4, seed=0
#   N_LIST="2 3 4"  bash experiment/sweep_insect_n.sh # custom n range
#   SEEDS="0 1 2"   bash experiment/sweep_insect_n.sh # multiple seeds per n
#   NUM_ENVS=2048   bash experiment/sweep_insect_n.sh # tighter VRAM budget
#   NUM_TIMESTEPS=20000000 bash experiment/sweep_insect_n.sh

set -euo pipefail

# Resolve script dir so relative paths work from any CWD.
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENT_DIR"

# Explicit interpreter from the nested mujoco_playground venv — avoids depending
# on the caller having activated the right venv or set VIRTUAL_ENV.
PY="$EXPERIMENT_DIR/mujoco_playground/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "error: expected mujoco_playground venv interpreter at $PY" >&2
    exit 1
fi

TRAIN_SCRIPT="rl/undirected_locomotion_ppo_jax.py"

# Overridable sweep parameters.
N_LIST="${N_LIST:-1 2 3 4}"
SEEDS="${SEEDS:-0 1 2}"
NUM_ENVS="${NUM_ENVS:-3072}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-200000000}"

total=0; for n in $N_LIST; do for s in $SEEDS; do total=$((total + 1)); done; done
echo "sweep: insect_n  n_pairs=[$N_LIST]  seeds=[$SEEDS]  (${total} run(s))"
echo "       --num-envs=$NUM_ENVS  --num-timesteps=$NUM_TIMESTEPS"
echo

i=0
for n in $N_LIST; do
    for s in $SEEDS; do
        i=$((i + 1))
        echo "===== [$i/$total] insect_n n_pairs=$n seed=$s ====="
        "$PY" "$TRAIN_SCRIPT" \
            --morphology insect_n \
            --n-pairs "$n" \
            --seed "$s" \
            --num-timesteps "$NUM_TIMESTEPS" \
            --num-envs "$NUM_ENVS"
        echo
    done
done

echo "sweep complete. runs in: __data__/undirected_locomotion_ppo_jax/insect_n_*"
