import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

# Local imports
from ariel.body_phenotypes.lynx_standalone.environment.table import TableWorld
from ariel.body_phenotypes.lynx_standalone.robot.constructor import construct_lynx

# --- UPGRADED NETWORK ARCHITECTURE ---
ACTION_SCALE = 1.2
CONTROL_DELTA_SCALE = 0.7
CTRL_FREQ = 20
GENOME_PATH = "best_lynx_brain_body.npy"
BRAIN_PATH = "best_lynx_brain_weights.npy"
MORPH_PATH = "best_lynx_morphology.npy"
SUCCESS_THRESHOLD_DEFAULT = 0.05
TARGETS = [
    [0.20, 0.00, 1.5],
]
GENOTYPE_TUBE = [1, 0, 0, 1, 0]
GENOTYPE_JOINTS = 6

# --- UPGRADED NETWORK ARCHITECTURE ---
# 3 (TCP) + 3 (Target) + 6 (Joint Angles) = 12 Inputs
INPUT_SIZE = 12 
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 6  # Lynx standalone has 6 joint actuators

# Define shapes for easy unflattening
W1_SHAPE = (INPUT_SIZE, HIDDEN_SIZE_1)
B1_SHAPE = (HIDDEN_SIZE_1,)
W2_SHAPE = (HIDDEN_SIZE_1, HIDDEN_SIZE_2)
B2_SHAPE = (HIDDEN_SIZE_2,)
W3_SHAPE = (HIDDEN_SIZE_2, OUTPUT_SIZE)
B3_SHAPE = (OUTPUT_SIZE,)

# Calculate exact parameter counts
W1_SIZE = int(np.prod(W1_SHAPE))
B1_SIZE = int(np.prod(B1_SHAPE))
W2_SIZE = int(np.prod(W2_SHAPE))
B2_SIZE = int(np.prod(B2_SHAPE))
W3_SIZE = int(np.prod(W3_SHAPE))
B3_SIZE = int(np.prod(B3_SHAPE))

# Total Genome Size
NUM_WEIGHTS = int(W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE)
TOTAL_GENOME_SIZE = int(11 + NUM_WEIGHTS) # 5 tubes + 6 rotations + weights

def simple_mlp(obs, weights):
    """Safely unflattens the 1D genome into 3 layers."""
    assert len(obs) == INPUT_SIZE, f"Shape Mismatch! Obs is {len(obs)}, expected {INPUT_SIZE}"
    assert len(weights) == NUM_WEIGHTS, f"Shape Mismatch! Weights is {len(weights)}, expected {NUM_WEIGHTS}"
    
    idx = 0
    W1 = weights[idx : idx + W1_SIZE].reshape(W1_SHAPE); idx += W1_SIZE
    b1 = weights[idx : idx + B1_SIZE].reshape(B1_SHAPE); idx += B1_SIZE
    
    W2 = weights[idx : idx + W2_SIZE].reshape(W2_SHAPE); idx += W2_SIZE
    b2 = weights[idx : idx + B2_SIZE].reshape(B2_SHAPE); idx += B2_SIZE
    
    W3 = weights[idx : idx + W3_SIZE].reshape(W3_SHAPE); idx += W3_SIZE
    b3 = weights[idx : idx + B3_SIZE].reshape(B3_SHAPE)
    
    # Forward Pass
    h1 = np.tanh(np.dot(obs, W1) + b1)
    h2 = np.tanh(np.dot(h1, W2) + b2)
    out = np.tanh(np.dot(h2, W3) + b3)
    
    return out
HOME_JOINT_ANGLES = np.zeros(6, dtype=np.float64)


def get_actuated_joint_ids(model: mujoco.MjModel, count: int = 6) -> list[int]:
    ids: list[int] = []
    for i in range(min(count, model.nu)):
        jid = int(model.actuator_trnid[i, 0])
        if jid >= 0:
            ids.append(jid)
    return ids


def resolve_site_id(model: mujoco.MjModel, base_name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, base_name)
    if sid != -1:
        return sid
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if name.endswith(base_name):
            return i
    return -1


def get_joint_angles(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    angles = np.zeros(6, dtype=np.float64)
    for i, jid in enumerate(get_actuated_joint_ids(model, count=6)):
        qaddr = model.jnt_qposadr[jid]
        angles[i] = data.qpos[qaddr]
    return angles


def apply_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    joint_ids = get_actuated_joint_ids(model, count=6)
    for i, jid in enumerate(joint_ids):
        qaddr = model.jnt_qposadr[jid]
        qval = float(HOME_JOINT_ANGLES[i])
        if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            qval = float(np.clip(qval, lo, hi))
        data.qpos[qaddr] = qval
        if i < model.nu:
            data.ctrl[i] = qval
    data.qvel[:] = 0.0


def load_artifacts() -> tuple[list[float], list[float], np.ndarray]:
    # Prefer explicit split files; fall back to combined genome for compatibility.
    if os.path.exists(BRAIN_PATH) and os.path.exists(MORPH_PATH):
        print("Loading split morphology + brain artifacts...")
        morph = np.load(MORPH_PATH)
        nn_weights = np.load(BRAIN_PATH)
        if len(morph) != 11:
            raise ValueError(f"Invalid morphology size: got {len(morph)}, expected 11")
        if len(nn_weights) != NUM_WEIGHTS:
            raise ValueError(
                f"Invalid brain weight size: got {len(nn_weights)}, expected {NUM_WEIGHTS}"
            )
        tube_lengths = np.clip(morph[0:5], 0.05, 0.4).tolist()
        rotations = np.clip(morph[5:11], -np.pi, np.pi).tolist()
        return tube_lengths, rotations, nn_weights

    print("Loading combined genome artifact...")
    genotype = np.load(GENOME_PATH)
    if len(genotype) != TOTAL_GENOME_SIZE:
        raise ValueError(
            f"Invalid genome size: got {len(genotype)}, expected {TOTAL_GENOME_SIZE}. "
            "Replay network/config does not match training setup."
        )
    tube_lengths = np.clip(genotype[0:5], 0.05, 0.4).tolist()
    rotations = np.clip(genotype[5:11], -np.pi, np.pi).tolist()
    nn_weights = genotype[11:]
    return tube_lengths, rotations, nn_weights


def build_model(tube_lengths: list[float], rotations: list[float]) -> tuple[mujoco.MjModel, mujoco.MjData, int, int]:
    robot_desc = {
        "num_joints": 6,
        "genotype_tube": GENOTYPE_TUBE,
        "genotype_joints": GENOTYPE_JOINTS,
        "tube_lengths": tube_lengths,
        "rotation_angles": rotations,
        "task": "reach"
    }
    world = TableWorld()
    robot = construct_lynx(robot_description_dict=robot_desc)
    world.spawn(robot.spec)

    model = world.spec.compile()
    data = mujoco.MjData(model)
    tcp_site_id = resolve_site_id(model, "end_effector_tcp")
    target_site_id = resolve_site_id(model, "target")
    if tcp_site_id == -1 or target_site_id == -1:
        raise RuntimeError("Could not resolve required TCP/target site IDs in replay model")
    return model, data, tcp_site_id, target_site_id


def run_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tcp_site_id: int,
    target_site_id: int,
    nn_weights: np.ndarray,
    target_position: list[float],
    sim_steps: int,
) -> tuple[float, float, float]:
    model.site_pos[target_site_id] = np.asarray(target_position, dtype=np.float64)
    apply_home_pose(model, data)
    mujoco.mj_forward(model, data)

    init_d = float(np.linalg.norm(data.site_xpos[tcp_site_id] - data.site_xpos[target_site_id]))
    min_d = init_d
    final_d = init_d

    for step in range(sim_steps):
        if step % (int(60 / CTRL_FREQ)) == 0:
            tcp_pos = data.site_xpos[tcp_site_id]
            target_p = data.site_xpos[target_site_id]
            joint_angles = get_joint_angles(model, data)
            rel_target = target_p - tcp_pos
            obs = np.concatenate([tcp_pos, rel_target, joint_angles])

            action = simple_mlp(obs, nn_weights) * ACTION_SCALE
            desired = joint_angles[:OUTPUT_SIZE] + (CONTROL_DELTA_SCALE * action)

            for i, jid in enumerate(get_actuated_joint_ids(model, count=OUTPUT_SIZE)):
                if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    desired[i] = np.clip(desired[i], lo, hi)

            data.ctrl[:OUTPUT_SIZE] = desired

        mujoco.mj_step(model, data)
        d = float(np.linalg.norm(data.site_xpos[tcp_site_id] - data.site_xpos[target_site_id]))
        final_d = d
        if d < min_d:
            min_d = d

    return init_d, min_d, final_d


def run_eval_only(
    tube_lengths: list[float],
    rotations: list[float],
    nn_weights: np.ndarray,
    sim_steps: int,
    success_threshold: float,
) -> None:
    print("Running headless evaluation across targets...")
    rows = []
    for target in TARGETS:
        model, data, tcp_site_id, target_site_id = build_model(tube_lengths, rotations)
        init_d, min_d, final_d = run_rollout(
            model, data, tcp_site_id, target_site_id, nn_weights, target, sim_steps
        )
        progress = max(0.0, (init_d - min_d) / max(init_d, 1e-8))
        success = min_d <= success_threshold
        rows.append((target, init_d, min_d, final_d, progress, success))

    print("target | init_d | min_d | final_d | progress_frac | success")
    for target, init_d, min_d, final_d, progress, success in rows:
        print(f"{target} | {init_d:.4f} | {min_d:.4f} | {final_d:.4f} | {progress:.3f} | {success}")

    succ = sum(1 for r in rows if r[5])
    print(f"success_rate={succ}/{len(rows)}={succ/len(rows):.3f}")
    print(f"mean_min_d={np.mean([r[2] for r in rows]):.4f}")
    print(f"mean_progress={np.mean([r[4] for r in rows]):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay or evaluate Lynx standalone policy")
    parser.add_argument("--eval-only", action="store_true", help="Run headless quantitative evaluation and exit")
    parser.add_argument("--sim-steps", type=int, default=600, help="Simulation steps per target")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=SUCCESS_THRESHOLD_DEFAULT,
        help="Distance threshold for success (lower is stricter)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    tube_lengths, rotations, nn_weights = load_artifacts()

    if args.eval_only:
        run_eval_only(
            tube_lengths=tube_lengths,
            rotations=rotations,
            nn_weights=nn_weights,
            sim_steps=args.sim_steps,
            success_threshold=args.success_threshold,
        )
        return

    print("Building MuJoCo Model...")
    model, data, tcp_site_id, target_site_id = build_model(tube_lengths, rotations)
    model.site_pos[target_site_id] = np.asarray(TARGETS[0], dtype=np.float64)
    apply_home_pose(model, data)
    mujoco.mj_forward(model, data)

    print("Launching Viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            # Update control at the same frequency as evolution
            if step % (int(60 / CTRL_FREQ)) == 0:
                # Get observation components
                tcp_pos = data.site_xpos[tcp_site_id]
                target_p = data.site_xpos[target_site_id]
                joint_angles = get_joint_angles(model, data)
                rel_target = target_p - tcp_pos
                
                # Combine into size-12 array
                obs = np.concatenate([tcp_pos, rel_target, joint_angles])
                
                # Get Action from Brain
                action = simple_mlp(obs, nn_weights) * ACTION_SCALE
                
                # RELATIVE POSITION CONTROL: Add action to CURRENT joint angles
                desired = joint_angles[:OUTPUT_SIZE] + (CONTROL_DELTA_SCALE * action)
                
                # Enforce basic limits
                for i, jid in enumerate(get_actuated_joint_ids(model, count=OUTPUT_SIZE)):
                    if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
                        lo, hi = model.jnt_range[jid]
                        desired[i] = np.clip(desired[i], lo, hi)

                data.ctrl[:OUTPUT_SIZE] = desired

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 60.0)
            step += 1

if __name__ == "__main__":
    main()