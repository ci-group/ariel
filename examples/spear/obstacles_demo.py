"""Visualize procedural obstacle layouts in a MuJoCo passive viewer.

Run:
    uv run examples/spear/obstacles_demo.py                  # forest
    uv run examples/spear/obstacles_demo.py --type urban
    uv run examples/spear/obstacles_demo.py --type indoor --num 30
    uv run examples/spear/obstacles_demo.py --type gates --no-view  # save PNG

NOTE: do NOT add `from __future__ import annotations` to this file.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ariel.simulation.environments import (
    ObstacleConfig,
    ObstacleType,
    SimpleFlatWorld,
    attach_obstacles,
    generate_obstacles,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="forest",
                        choices=[t.value for t in ObstacleType
                                 if t not in (ObstacleType.NONE, ObstacleType.CUSTOM)])
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--arena", type=float, nargs=3, default=(3.0, 3.0, 2.5),
                        metavar=("XH", "YH", "ZMAX"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--safe-radius", type=float, default=0.5)
    parser.add_argument("--no-view", action="store_true",
                        help="Render one PNG frame instead of opening the viewer.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (--no-view mode). Default: ./obstacles.png")
    args = parser.parse_args()

    cfg = ObstacleConfig(
        obstacle_type=ObstacleType(args.type),
        num_obstacles=args.num,
        arena_size=tuple(args.arena),
        seed=args.seed,
        safe_zone_radius=args.safe_radius,
        safe_zone_centers=np.array([[0.0, 0.0, 0.0]]),
    )
    obstacles = generate_obstacles(cfg)
    print(f"generated {len(obstacles)} obstacles ({args.type})")

    world = SimpleFlatWorld()
    attach_obstacles(world.spec, obstacles)
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"compiled model: ngeom={model.ngeom}, nbody={model.nbody}")

    if args.no_view:
        out = Path(args.out) if args.out else Path("obstacles.png")
        with mujoco.Renderer(model, width=960, height=720) as renderer:
            renderer.update_scene(data)
            frame = renderer.render()
        from PIL import Image
        Image.fromarray(frame).save(str(out))
        print(f"saved → {out}")
        return 0

    print("launching viewer — close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            viewer.sync()
            slack = 0.02 - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
