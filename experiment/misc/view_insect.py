"""Open a MuJoCo viewer for insect_n(n) or insect_longspine_n(n).

Spawns multiple windows (one per n) when run with --all; otherwise opens one.
Mirrors ``view_centipede.py`` for the insect scaling variants.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
# Bypass GNOME xsettings schema lookup — on some Wayland + GNOME setups the
# `org.gnome.settings-daemon.plugins.xsettings` schema is missing the
# `antialiasing` key, which raises a fatal GLib-GIO-ERROR during window
# creation and kills the viewer before it appears.
os.environ.setdefault("GSETTINGS_BACKEND", "memory")

import mujoco
import mujoco.viewer

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.insect import (
    insect_small,
    insect_n,
    insect_longspine_n,
)
from ariel.simulation.environments import SimpleFlatWorld


VARIANTS = ("insect_small", "insect_n", "insect_longspine_n")


def build(variant: str, n: int) -> mujoco.MjModel:
    world = SimpleFlatWorld(load_precompiled=False)
    if variant == "insect_small":
        body = insect_small()
    elif variant == "insect_n":
        body = insect_n(n)
    elif variant == "insect_longspine_n":
        body = insect_longspine_n(n)
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    world.spawn(body.spec, position=[0, 0, 0.1])
    m = world.spec.compile()
    m.opt.timestep = 0.005
    for i in range(m.ngeom):
        m.geom_contype[i] = 1
        m.geom_conaffinity[i] = 1
    return m


def view(variant: str, n: int) -> None:
    m = build(variant, n)
    d = mujoco.MjData(m)
    label = variant if variant == "insect_small" else f"{variant}(n={n})"
    print(f"[{label}] nbody={m.nbody} ngeom={m.ngeom} nu={m.nu}")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=list(VARIANTS),
        default="insect_n",
        help="Which insect factory to view.",
    )
    ap.add_argument("--n", type=int, help="Single n (n_segments) to view")
    ap.add_argument("--all", action="store_true",
                    help="Spawn one viewer per n in [n-min, n-max]")
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=6)
    args = ap.parse_args()

    if args.variant == "insect_small":
        view("insect_small", 0)
        return

    if args.n is not None:
        view(args.variant, args.n)
        return

    if args.all:
        procs = []
        for n in range(args.n_min, args.n_max + 1):
            p = subprocess.Popen(
                [sys.executable, __file__, "--variant", args.variant, "--n", str(n)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(p)
            print(f"spawned viewer for {args.variant}(n={n}) (pid={p.pid})")
        for p in procs:
            p.wait()
        return

    ap.error("pass --n <N> or --all (or --variant insect_small)")


if __name__ == "__main__":
    main()
