"""Stage 2: Evolve a vision-based steering controller.

Loads the frozen gait network from Stage 1 and evolves a small
steering network that takes vision features + battery level and
outputs (turn_signal, speed_signal, gait_enable).

The combined controller:
  steering(vision, battery) → turn, speed, enable
  gait(joints, phase, turn, speed) → joint_commands
  output = joint_commands × enable
"""
import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
from datetime import datetime
import time, argparse, cv2
import numpy as np
import mujoco
import torch
from torch import nn
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import NEProblem
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers.utils.data_get import (
    get_state_from_data as get_robot_state,
)
from ariel.utils.renderers import VideoRecorder
from baby_robot import baby_robot

# ─── Args ───
parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=200)
parser.add_argument("--population", type=int, default=50)
parser.add_argument("--dur", type=int, default=30)
parser.add_argument("--num-actors", type=int, default=1)
parser.add_argument("--gait-weights", type=str,
                    default="__data__/train_gait/gait_best.npy",
                    help="Path to Stage 1 gait weights")
args = parser.parse_args()
args.gait_weights = str(Path(args.gait_weights).resolve())

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population
REACH_RADIUS = 0.15

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA = Path("__data__/train_homing")
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "videos").mkdir(exist_ok=True)

# Targets at various distances up to 1.5m
TARGET_POSITIONS = [
    [0.0, -1.0, 0.1],
    [-0.7, -0.7, 0.1],
    [0.7, -0.7, 0.1],
    [0.0, -1.5, 0.1],
    [-1.0, -1.0, 0.1],
    [1.0, -1.0, 0.1],
]

BATTERY_THRESHOLD = 0.3


# ─── Gait Network (frozen, loaded from Stage 1) ───
class GaitNetwork(nn.Module):
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


def load_frozen_gait(weights_path, input_dim, output_dim, hidden_size=16):
    """Load Stage 1 gait weights into a frozen GaitNetwork."""
    net = GaitNetwork(input_dim, output_dim, hidden_size)
    weights = np.load(weights_path)
    vec = torch.tensor(weights, dtype=torch.float32)
    addr = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = vec[addr:addr+n]
        addr += n
    return net


# ─── Steering Network (small, this is what we evolve) ───
class SteeringNetwork(nn.Module):
    """Tiny network: vision + battery → turn, speed, gait_enable.
    
    Inputs (8):
        - vision features (7): 5 strip percentages + centroid_x + area_fraction
        - battery (1)
    
    Outputs (3):
        - turn_signal [-1, +1]: fed to gait network
        - speed_signal [0, 1]: fed to gait network
        - gait_enable [0, 1]: multiplied with gait output
    """
    def __init__(self, hidden_size=8):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden_size)   # 8 inputs
        self.fc_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 3)   # 3 outputs
        self.hidden_size = hidden_size
        self._h = None
        for p in self.parameters():
            p.requires_grad = False

    def reset_hidden(self):
        self._h = torch.zeros(self.hidden_size)

    @torch.inference_mode()
    def forward(self, vision_features, battery):
        x = torch.tensor(
            np.concatenate([vision_features, [battery]]),
            dtype=torch.float32
        )
        if self._h is None:
            self.reset_hidden()
        h = torch.nn.functional.elu(self.fc1(x) + self.fc_rec(self._h))
        self._h = torch.tanh(h).detach().clone()
        raw = self.fc2(h)
        
        turn = torch.tanh(raw[0]).item()           # [-1, +1]
        speed = torch.sigmoid(raw[1]).item()        # [0, 1]
        enable = torch.sigmoid(raw[2]).item()       # [0, 1]
        
        return turn, speed, enable


# ─── Vision processing (same as before) ───
def isolate_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 30, 30])
    upper = np.array([95, 255, 255])
    return cv2.inRange(hsv, lower, upper)

def analyze_sections(green_mask):
    h, w = green_mask.shape
    sections = np.array_split(green_mask, 5, axis=1)
    strips = []
    for s in sections:
        total = s.size
        strips.append(float(cv2.countNonZero(s)) / total if total > 0 else 0.0)
    total_green = cv2.countNonZero(green_mask)
    if total_green > 0:
        m = cv2.moments(green_mask)
        centroid_x = (m["m10"] / m["m00"] / w) * 2.0 - 1.0
    else:
        centroid_x = 0.0
    area = float(total_green) / float(h * w)
    return strips + [centroid_x, area]


# ─── Combined simulation runner ───
def run_homing_episode(model, data, gait_net, steer_net, duration,
                       target_pos, renderer=None, cam_name=None):
    """Run one homing episode with the hierarchical controller."""
    gait_net.reset_hidden()
    steer_net.reset_hidden()
    
    timestep = model.opt.timestep
    control_freq = 50
    current_action = np.zeros(model.nu)
    
    battery = 1.0
    drain = timestep / max(float(duration), 1.0)
    
    trajectory = []
    shaped_homing = 0.0
    prev_dist = float(np.linalg.norm(data.qpos[:2] - np.asarray(target_pos)[:2]))
    time_to_target = None
    
    # Create local renderer if needed
    created_renderer = False
    if renderer is None:
        try:
            renderer = mujoco.Renderer(model, height=24, width=32)
            created_renderer = True
        except Exception:
            renderer = None
    
    while data.time < duration:
        step = int(np.ceil(data.time / timestep))
        
        if step % control_freq == 0:
            # ── Vision ──
            if renderer is not None:
                try:
                    renderer.update_scene(data, camera=cam_name)
                    img = renderer.render()
                    vision = analyze_sections(isolate_green(img))
                except Exception:
                    vision = [0.0] * 7
            else:
                vision = [0.0] * 7
            
            # ── Steering network: vision + battery → turn, speed, enable ──
            turn, speed, enable = steer_net.forward(vision, battery)
            
            # ── Gait network: joints + phase + turn + speed → joint cmds ──
            robot_state = get_robot_state(data)
            phase = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]
            gait_input = np.concatenate([
                robot_state, phase, [turn, speed]
            ]).astype(np.float32)
            
            joint_commands = gait_net.forward(gait_input)
            
            # ── Combine: gait × enable ──
            current_action = joint_commands * enable
            
            if not np.all(np.isfinite(current_action)):
                current_action = np.zeros(model.nu)
            
            trajectory.append((data.qpos[0], data.qpos[1], battery))
        
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        battery = max(0.0, battery - drain)
        
        # Track homing progress
        cur_dist = float(np.linalg.norm(data.qpos[:2] - np.asarray(target_pos)[:2]))
        if battery <= BATTERY_THRESHOLD:
            delta = prev_dist - cur_dist
            if delta > 0:
                shaped_homing += delta
        if time_to_target is None and cur_dist <= REACH_RADIUS:
            time_to_target = float(data.time)
        prev_dist = cur_dist
    
    if created_renderer:
        try:
            renderer.close()
        except Exception:
            pass
    
    final_pos = data.qpos[:2].copy()
    final_dist = float(np.linalg.norm(final_pos - np.asarray(target_pos)[:2]))
    
    return {
        "final_dist": final_dist,
        "shaped_homing": shaped_homing,
        "time_to_target": time_to_target,
        "final_z": float(data.qpos[2]),
        "trajectory": trajectory,
    }


# ─── Actor environment ───
_HOMING_ENV: dict = {}

def _init_homing_env():
    if _HOMING_ENV:
        return
    
    world = SimpleFlatWorld()
    target_body = world.spec.worldbody.add_body(
        name="charging_station", mocap=True, pos=TARGET_POSITIONS[0])
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1], rgba=[0, 1, 0, 1])
    world.spec.worldbody.add_camera(
        name="video_cam", pos=[0, -1, 3], xyaxes=[1, 0, 0, 0, 3, 0])
    
    baby_core = baby_robot()
    world.spawn(baby_core.spec, position=[0, 0, 0.1])
    
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    # Find robot camera
    cam_name = None
    for i in range(model.ncam):
        n = model.camera(i).name
        if ("camera" in n or "core" in n) and "video" not in n:
            cam_name = n
            break
    
    try:
        renderer = mujoco.Renderer(model, height=24, width=32)
    except Exception:
        renderer = None
    
    mocap_id = model.body("charging_station").mocapid[0]
    
    # Load frozen gait network
    meta = np.load(str(Path("__data__/train_gait/gait_meta.npz").resolve()))
    gait_net = load_frozen_gait(
        args.gait_weights,
        int(meta["input_dim"]),
        int(meta["output_dim"]),
        int(meta["hidden_size"]),
    )
    
    _HOMING_ENV.update({
        "model": model, "data": data,
        "cam_name": cam_name, "renderer": renderer,
        "mocap_id": mocap_id, "gait_net": gait_net,
    })


def homing_fitness(steer_net) -> float:
    """Evaluate the steering network with frozen gait."""
    _init_homing_env()
    
    model = _HOMING_ENV["model"]
    data = _HOMING_ENV["data"]
    gait_net = _HOMING_ENV["gait_net"]
    renderer = _HOMING_ENV["renderer"]
    cam_name = _HOMING_ENV["cam_name"]
    mocap_id = _HOMING_ENV["mocap_id"]
    
    total = 0.0
    for target in TARGET_POSITIONS:
        mujoco.mj_resetData(model, data)
        data.mocap_pos[mocap_id] = target
        
        metrics = run_homing_episode(
            model, data, gait_net, steer_net,
            DURATION, target, renderer, cam_name,
        )
        
        final_dist = metrics["final_dist"]
        homing = -float(metrics["shaped_homing"])
        arrival = -10.0 if metrics["time_to_target"] is not None else 0.0
        flip = 5.0 if metrics["final_z"] < 0.02 else 0.0
        
        score = 5.0 * final_dist + 2.0 * homing + arrival + flip
        total += score
    
    return total / len(TARGET_POSITIONS)


def main():
    # Verify Stage 1 weights exist
    if not os.path.exists(args.gait_weights):
        print(f"ERROR: Gait weights not found at {args.gait_weights}")
        print("Run train_gait.py (Stage 1) first!")
        return
    
    steer_net = SteeringNetwork(hidden_size=8)
    n_params = sum(p.numel() for p in steer_net.parameters())
    print(f"Steering network: 8 inputs, 3 outputs, {n_params} params")
    print(f"(Compare: your old monolithic network had ~750 params)")
    
    num_actors_cfg = args.num_actors if args.num_actors > 1 else None
    actor_config = None
    if num_actors_cfg:
        actor_config = {
            "num_cpus": 1,
            "runtime_env": {"excludes": [".git", ".venv", "__pycache__"]},
        }
    
    problem = NEProblem(
        objective_sense="min",
        network_eval_func=homing_fitness,
        network=steer_net.eval(),
        initial_bounds=(-1.0, 1.0),
        device="cpu",
        num_actors=num_actors_cfg,
        actor_config=actor_config,
    )
    
    searcher = CMAES(problem=problem, stdev_init=0.5, popsize=POP_SIZE)
    print(f"Pop size: {searcher.popsize}")
    
    history = []
    for gen in range(BUDGET + 1):
        searcher.step()
        best = float(searcher.status["pop_best_eval"])
        history.append(best)
        if gen % 10 == 0:
            print(f"Gen {gen}/{BUDGET} — Best: {best:.4f}")
        if gen > 0 and gen % 25 == 0:
            np.save(str(DATA / f"steer_ckpt_gen{gen}.npy"),
                    searcher.status["best"].values.numpy())
    
    # Save
    best_weights = searcher.status["best"].values.numpy()
    np.save(str(DATA / f"steer_best_{RUN_TIMESTAMP}.npy"), best_weights)
    
    # Save fitness history
    np.save(str(DATA / "fitness_history.npy"), np.array(history))
    
    # Plot fitness
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, "b-", linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower = better)")
    ax.set_title("Stage 2: Homing Steering Evolution")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(DATA / f"homing_fitness_{RUN_TIMESTAMP}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return best_weights


if __name__ == "__main__":
    start = time.time()
    best_weights = main()
    if best_weights is None:
        exit(1)
    print(f"Stage 2 took {(time.time() - start) / 60:.1f} minutes")
    
    # ─── Replay & record video ───
    print("Recording demo video...")
    
    _init_homing_env()
    model = _HOMING_ENV["model"]
    data = _HOMING_ENV["data"]
    gait_net = _HOMING_ENV["gait_net"]
    cam_name = _HOMING_ENV["cam_name"]
    mocap_id = _HOMING_ENV["mocap_id"]
    
    # Load best steering weights
    steer_net = SteeringNetwork(hidden_size=8)
    vec = torch.tensor(best_weights, dtype=torch.float32)
    addr = 0
    for p in steer_net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = vec[addr:addr+n]
        addr += n
    
    # Demo target
    demo_target = [0.0, -1.0, 0.1]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[mocap_id] = demo_target
    
    # Run evaluation for trajectory plot
    try:
        plot_renderer = mujoco.Renderer(model, height=24, width=32)
    except Exception:
        plot_renderer = None
    
    metrics = run_homing_episode(
        model, data, gait_net, steer_net,
        DURATION, demo_target, plot_renderer, cam_name,
    )
    
    if plot_renderer:
        try:
            plot_renderer.close()
        except Exception:
            pass
    
    # ── Trajectory + Battery plot ──
    path = metrics["trajectory"]
    if len(path) > 1:
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])
        b = np.array([p[2] for p in path])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cmap = LinearSegmentedColormap.from_list("b", ["red", "orange", "green"])
        lc = LineCollection(segs, cmap=cmap, linewidth=2.5)
        lc.set_array(b[:-1])
        ax1.add_collection(lc)
        ax1.plot(x[0], y[0], "o", color="green", markersize=12, label="Start", zorder=5)
        ax1.plot(demo_target[0], demo_target[1], "*", color="red",
                 markersize=18, label="Charging Station", zorder=5)
        ax1.autoscale()
        ax1.set_aspect("equal")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Trajectory (coloured by battery)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig.colorbar(lc, ax=ax1, label="Battery")
        
        t = np.linspace(0, DURATION, len(path))
        d = np.sqrt((x - demo_target[0])**2 + (y - demo_target[1])**2)
        ax2.plot(t, b, "g-", lw=2, label="Battery")
        ax2.axhline(y=0.3, color="orange", ls="--", alpha=0.7, label="Threshold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Battery", color="green")
        ax2r = ax2.twinx()
        ax2r.plot(t, d, "r-", lw=2, label="Distance")
        ax2r.set_ylabel("Distance (m)", color="red")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2)
        ax2.set_title("Battery & Distance Over Time")
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(str(DATA / f"trajectory_{RUN_TIMESTAMP}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Trajectory plot saved.")
    
    # ── Video recording ──
    mujoco.mj_resetData(model, data)
    gait_net.reset_hidden()
    steer_net.reset_hidden()
    data.mocap_pos[mocap_id] = demo_target
    
    try:
        vid_renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        vid_renderer = None
    
    try:
        ctrl_renderer = mujoco.Renderer(model, height=24, width=32)
    except Exception:
        ctrl_renderer = None
    
    if vid_renderer is not None:
        video_recorder = VideoRecorder(
            file_name=f"homing_demo_{RUN_TIMESTAMP}",
            output_folder=str(DATA / "videos"),
        )
        
        dt = model.opt.timestep
        fps = 30
        steps_per_frame = int(1.0 / (fps * dt))
        control_freq = 50
        current_ctrl = np.zeros(model.nu)
        battery = 1.0
        drain = dt / max(float(DURATION), 1.0)
        
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
        viz = mujoco.MjvOption()
        viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        
        while data.time < DURATION:
            for _ in range(steps_per_frame):
                step = int(np.ceil(data.time / dt))
                if step % control_freq == 0:
                    # Vision
                    if ctrl_renderer:
                        try:
                            ctrl_renderer.update_scene(data, camera=cam_name)
                            img = ctrl_renderer.render()
                            vision = analyze_sections(isolate_green(img))
                        except Exception:
                            vision = [0.0] * 7
                    else:
                        vision = [0.0] * 7
                    
                    turn, speed, enable = steer_net.forward(vision, battery)
                    robot_state = get_robot_state(data)
                    phase = [
                        2 * np.sin(data.time * 2 * np.pi),
                        2 * np.cos(data.time * 2 * np.pi),
                    ]
                    gait_in = np.concatenate([
                        robot_state, phase, [turn, speed]
                    ]).astype(np.float32)
                    current_ctrl = gait_net.forward(gait_in) * enable
                    if not np.all(np.isfinite(current_ctrl)):
                        current_ctrl = np.zeros(model.nu)
                
                data.ctrl[:] = current_ctrl
                mujoco.mj_step(model, data)
                battery = max(0.0, battery - drain)
            
            vid_renderer.update_scene(data, scene_option=viz, camera=camera_id)
            video_recorder.write(frame=vid_renderer.render())
        
        video_recorder.release()
        vid_renderer.close()
        print(f"Video saved → {DATA / 'videos'}")
    
    if ctrl_renderer:
        try:
            ctrl_renderer.close()
        except Exception:
            pass
    
    print("Done.")