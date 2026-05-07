"""Stage 1: Evolve a gait controller for the Baby robot.

The gait network takes joint state + phase + turn/speed signals
and produces joint commands. It is evolved with CMA-ES to walk
toward target positions placed at different angles.
"""
import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import time, argparse
import numpy as np
import mujoco
import torch
from torch import nn
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import NEProblem
import matplotlib.pyplot as plt

from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers.utils.data_get import (
    get_state_from_data as get_robot_state,
)
from baby_robot import baby_robot

# ─── Args ───
parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=300)
parser.add_argument("--population", type=int, default=50)
parser.add_argument("--dur", type=int, default=10)
parser.add_argument("--num-actors", type=int, default=1)
args = parser.parse_args()

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population

DATA = Path("__data__/train_gait")
DATA.mkdir(parents=True, exist_ok=True)

# Target positions: different directions at ~1m distance
# The gait network receives a "turn signal" derived from the
# angle to the target, so it must learn to steer.
GAIT_TARGETS = [
    [0.0, -1.0, 0.1],     # straight ahead
    [-0.7, -0.7, 0.1],    # 45° left
    [0.7, -0.7, 0.1],     # 45° right
    [-1.0, 0.0, 0.1],     # 90° left
    [1.0, 0.0, 0.1],      # 90° right
]


# ─── Gait Network ───
class GaitNetwork(nn.Module):
    """Small recurrent network for locomotion.
    
    Inputs:
        - robot_state: joint angles + orientation (from get_robot_state)
        - phase: [sin(t), cos(t)] for gait rhythm
        - turn_signal: [-1, +1] desired turning direction
        - speed_signal: [0, 1] desired speed (1 = full, 0 = stop)
    
    Outputs:
        - joint commands (model.nu values)
    """
    def __init__(self, input_size, output_size, hidden_size=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self._h = None
        for p in self.parameters():
            p.requires_grad = False

    def reset_hidden(self):
        self._h = torch.zeros(self.hidden_size)

    @torch.inference_mode()
    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        if self._h is None:
            self.reset_hidden()
        h = torch.nn.functional.elu(self.fc1(x) + self.fc_rec(self._h))
        h = torch.nn.functional.elu(self.fc2(h))
        self._h = torch.tanh(h).detach().clone()
        return (torch.tanh(self.fc_out(h)) * (torch.pi / 2)).detach().numpy()


def compute_turn_signal(robot_pos, robot_quat, target_pos):
    """Compute a [-1, +1] turn signal: which direction to turn to face target.
    
    Uses the robot's yaw (from quaternion) and the angle to the target.
    +1 = target is to the right, -1 = target is to the left.
    """
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    target_angle = np.arctan2(dy, dx)
    
    # Extract yaw from quaternion (quat = [w, x, y, z])
    w, x, y, z = robot_quat
    robot_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Angle difference, normalized to [-pi, pi]
    angle_diff = target_angle - robot_yaw
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Clamp to [-1, 1]
    return float(np.clip(angle_diff / np.pi, -1.0, 1.0))


def run_gait_episode(model, data, network, duration, target_pos):
    """Run one episode of gait training. Returns distance to target at end."""
    network.reset_hidden()
    timestep = model.opt.timestep
    control_freq = 50
    current_action = np.zeros(model.nu)

    while data.time < duration:
        step = int(np.ceil(data.time / timestep))
        if step % control_freq == 0:
            robot_state = get_robot_state(data)
            phase = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]
            
            # Compute turn signal from known target position
            # (In Stage 1 the robot "knows" where the target is —
            #  this is NOT cheating because Stage 2 will replace
            #  this signal with one derived from vision.)
            robot_pos = data.qpos[:3].copy()
            robot_quat = data.qpos[3:7].copy()
            turn = compute_turn_signal(robot_pos, robot_quat, target_pos)
            speed = 1.0  # always full speed during gait training
            
            state = np.concatenate([
                robot_state, phase, [turn, speed]
            ]).astype(np.float32)
            
            current_action = network.forward(state)
            if not np.all(np.isfinite(current_action)):
                current_action = np.zeros(model.nu)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

    final_pos = data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(final_pos - np.asarray(target_pos)[:2]))
    final_z = float(data.qpos[2])
    
    return final_dist, final_z


# ─── Actor-local environment ───
_GAIT_ENV: dict = {}

def _init_gait_env():
    if _GAIT_ENV:
        return
    world = SimpleFlatWorld()
    
    # No charging station needed — just the robot and a flat arena
    baby_core = baby_robot()
    world.spawn(baby_core.spec, position=[0, 0, 0.1])
    
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    _GAIT_ENV.update({"model": model, "data": data})


def gait_fitness(net) -> float:
    """Fitness: average final distance across all target directions."""
    _init_gait_env()
    model = _GAIT_ENV["model"]
    data = _GAIT_ENV["data"]
    
    total = 0.0
    for target in GAIT_TARGETS:
        mujoco.mj_resetData(model, data)
        final_dist, final_z = run_gait_episode(
            model, data, net, DURATION, target
        )
        
        flip_penalty = 5.0 if final_z < 0.02 else 0.0
        total += 5.0 * final_dist + flip_penalty
    
    return total / len(GAIT_TARGETS)


def main():
    # Build a temporary model to get dimensions
    world = SimpleFlatWorld()
    baby_core = baby_robot()
    world.spawn(baby_core.spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    num_joints = len(data.qpos) - 7
    # Inputs: robot_state(3 + num_joints) + phase(2) + turn(1) + speed(1)
    input_dim = 3 + num_joints + 2 + 1 + 1
    
    network = GaitNetwork(input_dim, model.nu, hidden_size=16)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"Gait network: {input_dim} inputs, {model.nu} outputs, {n_params} params")
    
    # Configure parallelism
    num_actors_cfg = args.num_actors if args.num_actors > 1 else None
    actor_config = None
    if num_actors_cfg:
        actor_config = {
            "num_cpus": 1,
            "runtime_env": {"excludes": [".git", ".venv", "__pycache__"]},
        }
    
    problem = NEProblem(
        objective_sense="min",
        network_eval_func=gait_fitness,
        network=network.eval(),
        initial_bounds=(-0.5, 0.5),
        device="cpu",
        num_actors=num_actors_cfg,
        actor_config=actor_config,
    )
    
    searcher = CMAES(problem=problem, stdev_init=0.3, popsize=POP_SIZE)
    print(f"Pop size: {searcher.popsize}")
    
    history = []
    for gen in range(BUDGET + 1):
        searcher.step()
        best = float(searcher.status["pop_best_eval"])
        history.append(best)
        if gen % 10 == 0:
            print(f"Gen {gen}/{BUDGET} — Best: {best:.4f}")
        if gen > 0 and gen % 50 == 0:
            np.save(str(DATA / f"gait_ckpt_gen{gen}.npy"),
                    searcher.status["best"].values.numpy())
    
    # Save best weights
    best_weights = searcher.status["best"].values.numpy()
    np.save(str(DATA / "gait_best.npy"), best_weights)
    print(f"Saved gait weights → {DATA / 'gait_best.npy'}")
    
    # Save metadata so Stage 2 knows the architecture
    np.savez(str(DATA / "gait_meta.npz"),
             input_dim=input_dim,
             output_dim=model.nu,
             hidden_size=16)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, "b-", linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower = better)")
    ax.set_title("Stage 1: Gait Evolution")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(DATA / "gait_fitness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Stage 1 took {(time.time() - start) / 60:.1f} minutes")

    # ─── Demo video for Stage 1 ───
    print("Recording Stage 1 demo video...")

    from ariel.utils.renderers import VideoRecorder
    from datetime import datetime

    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    (DATA / "videos").mkdir(exist_ok=True)

    # Rebuild environment with a visible target marker
    world = SimpleFlatWorld()

    # Add a visual target marker (not a charger, just a goal point)
    demo_target = [0.0, -0.8, 0.1]
    target_body = world.spec.worldbody.add_body(
        name="target_marker", mocap=True, pos=demo_target)
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.05, 0, 0],
        rgba=[1, 0, 0, 1],  # red sphere
    )

    # Add overhead camera
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, -0.5, 2.0],
        xyaxes=[1, 0, 0, 0, 2, 0],
    )

    baby_core = baby_robot()
    world.spawn(baby_core.spec, position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Load best gait weights
    meta = np.load(str(DATA / "gait_meta.npz"))
    input_dim = int(meta["input_dim"])
    output_dim = int(meta["output_dim"])
    hidden_size = int(meta["hidden_size"])

    gait_net = GaitNetwork(input_dim, output_dim, hidden_size)
    weights = np.load(str(DATA / "gait_best.npy"))
    vec = torch.tensor(weights, dtype=torch.float32)
    addr = 0
    for p in gait_net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = vec[addr:addr + n]
        addr += n

    # Setup video
    try:
        vid_renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        vid_renderer = None
        print("Video renderer unavailable, skipping.")

    if vid_renderer is not None:
        video_recorder = VideoRecorder(
            file_name=f"gait_demo_{RUN_TIMESTAMP}",
            output_folder=str(DATA / "videos"),
        )

        mujoco.mj_resetData(model, data)
        gait_net.reset_hidden()

        # Place target marker
        mocap_id = model.body("target_marker").mocapid[0]
        data.mocap_pos[mocap_id] = demo_target

        dt = model.opt.timestep
        fps = 30
        steps_per_frame = int(1.0 / (fps * dt))
        control_freq = 50
        current_ctrl = np.zeros(model.nu)
        duration = 15  # match training duration

        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
        viz = mujoco.MjvOption()
        viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

        while data.time < duration:
            for _ in range(steps_per_frame):
                step = int(np.ceil(data.time / dt))
                if step % control_freq == 0:
                    robot_state = get_robot_state(data)
                    phase = [
                        2 * np.sin(data.time * 2.0 * np.pi),
                        2 * np.cos(data.time * 2.0 * np.pi),
                    ]
                    robot_pos = data.qpos[:3].copy()
                    robot_quat = data.qpos[3:7].copy()
                    turn = compute_turn_signal(robot_pos, robot_quat, demo_target)
                    speed = 1.0

                    state = np.concatenate([
                        robot_state, phase, [turn, speed]
                    ]).astype(np.float32)

                    current_ctrl = gait_net.forward(state)
                    if not np.all(np.isfinite(current_ctrl)):
                        current_ctrl = np.zeros(model.nu)

                data.ctrl[:] = current_ctrl
                mujoco.mj_step(model, data)

            vid_renderer.update_scene(data, scene_option=viz, camera=camera_id)
            video_recorder.write(frame=vid_renderer.render())

        video_recorder.release()
        vid_renderer.close()
        print(f"Gait demo video saved → {DATA / 'videos'}")

    # Also save a trajectory plot for Stage 1
    mujoco.mj_resetData(model, data)
    gait_net.reset_hidden()
    data.mocap_pos[mocap_id] = demo_target
    traj = []
    current_ctrl = np.zeros(model.nu)

    while data.time < duration:
        step = int(np.ceil(data.time / dt))
        if step % control_freq == 0:
            robot_state = get_robot_state(data)
            phase = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]
            turn = compute_turn_signal(
                data.qpos[:3].copy(), data.qpos[3:7].copy(), demo_target)
            state = np.concatenate([
                robot_state, phase, [turn, 1.0]
            ]).astype(np.float32)
            current_ctrl = gait_net.forward(state)
            if not np.all(np.isfinite(current_ctrl)):
                current_ctrl = np.zeros(model.nu)
            traj.append((data.qpos[0], data.qpos[1]))
        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)

    if traj:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(xs, ys, "b-", linewidth=2, label="Robot path")
        ax.plot(xs[0], ys[0], "go", markersize=12, label="Start")
        ax.plot(demo_target[0], demo_target[1], "r*",
                markersize=18, label="Target")
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Stage 1: Gait Demo Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(str(DATA / f"gait_trajectory_{RUN_TIMESTAMP}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Gait trajectory plot saved.")