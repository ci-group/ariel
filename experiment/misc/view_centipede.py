"""Open a MuJoCo viewer for body_centipede_n(n).

Spawns multiple windows (one per n) when run with --all; otherwise opens one.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.centipede import body_centipede_n
from ariel.simulation.environments import SimpleFlatWorld


def build(n_pairs: int) -> mujoco.MjModel:
    world = SimpleFlatWorld(load_precompiled=False)
    body = body_centipede_n(n_pairs)
    world.spawn(body.spec, position=[0, 0, 0.1])
    m = world.spec.compile()
    m.opt.timestep = 0.005
    for i in range(m.ngeom):
        m.geom_contype[i] = 1
        m.geom_conaffinity[i] = 1
    return m


def view(n: int) -> None:
    m = build(n)
    d = mujoco.MjData(m)
    print(f"[n_pairs={n}] nbody={m.nbody} ngeom={m.ngeom} nu={m.nu}")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="Single n_pairs to view")
    ap.add_argument("--all", action="store_true",
                    help="Spawn one viewer per n in [n-min, n-max]")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=6)
    args = ap.parse_args()

    if args.n is not None:
        view(args.n)
        return

    if args.all:
        procs = []
        for n in range(args.n_min, args.n_max + 1):
            p = subprocess.Popen(
                [sys.executable, __file__, "--n", str(n)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(p)
            print(f"spawned viewer for n_pairs={n} (pid={p.pid})")
        for p in procs:
            p.wait()
        return

    ap.error("pass --n <N> or --all")


if __name__ == "__main__":
    main()
