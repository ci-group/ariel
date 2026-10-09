"""Sweep body_centipede_n(n_pairs) and report geom/contact/constraint scaling.

Compares the measured peaks against the per-env budgets hard-coded in
`experiment/rl/undirected_locomotion_ppo_jax.py`:
    NJMAX_PER_ENV    = 800
    NACONMAX_PER_ENV = 320
    NACCDMAX_PER_ENV = 100
These were tuned for n_pairs=2 (the `insect_small` default). This script lets
you see at what n_pairs they start to be tight.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse

import numpy as np
import mujoco

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.centipede import body_centipede_n
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.insect import insect_small
from ariel.simulation.environments import SimpleFlatWorld


NJMAX_PER_ENV = 800
NACONMAX_PER_ENV = 320
NACCDMAX_PER_ENV = 100


def build(label):
    """label: int n_pairs, or the literal string 'insect_small'."""
    world = SimpleFlatWorld(load_precompiled=False)
    body = insect_small() if label == "insect_small" else body_centipede_n(label)
    world.spawn(body.spec, position=[0, 0, 0.1])
    m = world.spec.compile()
    m.opt.timestep = 0.005
    m.opt.iterations = 6
    m.opt.ls_iterations = 12
    for i in range(m.ngeom):
        m.geom_contype[i] = 1
        m.geom_conaffinity[i] = 1
    floor_id = m.geom("floor").id
    return m, floor_id


def count_floor_pairs(m, floor_id):
    n = 0
    for i in range(m.ngeom):
        if i == floor_id:
            continue
        if (m.geom_contype[i] & m.geom_conaffinity[floor_id]) or (
            m.geom_contype[floor_id] & m.geom_conaffinity[i]
        ):
            n += 1
    return n


def runtime_stats(m, steps: int = 500):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    peak_ncon = 0
    peak_nefc = 0
    for s in range(steps):
        d.ctrl[:] = 0.5 * np.sin(0.1 * s + np.arange(m.nu))
        mujoco.mj_step(m, d)
        peak_ncon = max(peak_ncon, int(d.ncon))
        peak_nefc = max(peak_nefc, int(d.nefc))
    return peak_ncon, peak_nefc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    header = (
        f"{'morph':>14} {'nbody':>6} {'ngeom':>6} {'njnt':>5} {'nu':>4} {'nv':>4} "
        f"{'cand_floor':>10} {'peak_ncon':>10} {'peak_nefc':>10} "
        f"{'ncon/320':>9} {'nefc/800':>9}"
    )
    print(header)
    print("-" * len(header))
    labels = ["insect_small"] + list(range(1, args.n_max + 1))
    for label in labels:
        m, floor_id = build(label)
        cfp = count_floor_pairs(m, floor_id)
        p_ncon, p_nefc = runtime_stats(m, steps=args.steps)
        pretty = label if isinstance(label, str) else f"centipede_n={label}"
        print(
            f"{pretty:>14} {m.nbody:>6} {m.ngeom:>6} {m.njnt:>5} {m.nu:>4} {m.nv:>4} "
            f"{cfp:>10} {p_ncon:>10} {p_nefc:>10} "
            f"{p_ncon / NACONMAX_PER_ENV:>9.2f} {p_nefc / NJMAX_PER_ENV:>9.2f}"
        )


if __name__ == "__main__":
    main()
