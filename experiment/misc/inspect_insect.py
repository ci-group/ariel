"""Count geoms, floor-contact pairs, and peak constraint rows for the insect."""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.centipede import body_centipede as insect_small
from ariel.simulation.environments import SimpleFlatWorld


def build():
    world = SimpleFlatWorld(load_precompiled=False)
    body = insect_small()
    world.spawn(body.spec, position=[0, 0, 0.1])
    m = world.spec.compile()
    m.opt.timestep = 0.005
    m.opt.iterations = 4
    m.opt.ls_iterations = 8

    floor_id = m.geom("floor").id
    for i in range(m.ngeom):
        if i == floor_id:
            m.geom_contype[i] = 1
            m.geom_conaffinity[i] = 2
        else:
            m.geom_contype[i] = 2
            m.geom_conaffinity[i] = 1
    return m, floor_id


def count_floor_pairs(m, floor_id):
    """Enumerate how many non-floor geoms could ever touch the floor
    (contype/conaffinity respected)."""
    n = 0
    for i in range(m.ngeom):
        if i == floor_id:
            continue
        if (m.geom_contype[i] & m.geom_conaffinity[floor_id]) or (
            m.geom_contype[floor_id] & m.geom_conaffinity[i]
        ):
            n += 1
    return n


def runtime_stats(m):
    """Simulate briefly, record peak ncon / nefc."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    peak_ncon = 0
    peak_nefc = 0
    for s in range(500):  # 2.5 s of settle + random control
        d.ctrl[:] = 0.5 * np.sin(0.1 * s + np.arange(m.nu))
        mujoco.mj_step(m, d)
        peak_ncon = max(peak_ncon, d.ncon)
        peak_nefc = max(peak_nefc, d.nefc)
    return peak_ncon, peak_nefc


def main():
    m, floor_id = build()
    print(f"nbody         = {m.nbody}")
    print(f"ngeom         = {m.ngeom}")
    print(f"njnt          = {m.njnt}")
    print(f"nu (actuators)= {m.nu}")
    print(f"nv (DoF)      = {m.nv}")
    print(f"candidate floor-contact pairs = {count_floor_pairs(m, floor_id)}")
    peak_ncon, peak_nefc = runtime_stats(m)
    print(f"peak CPU ncon (500 steps) = {peak_ncon}")
    print(f"peak CPU nefc (500 steps) = {peak_nefc}")


if __name__ == "__main__":
    main()
