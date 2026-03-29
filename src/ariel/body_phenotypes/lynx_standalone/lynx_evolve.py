import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import nevergrad as ng
from rich.console import Console

# Local imports from your standalone setup
from ariel.body_phenotypes.lynx_standalone.environment.table import TableWorld
from ariel.body_phenotypes.lynx_standalone.robot.constructor import construct_lynx

console = Console()

DATA_DIR = Path.cwd() / "__data__" / "lynx_standalone"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Evolution settings
NUM_GENERATIONS = 40
POP_SIZE = 16
SIM_STEPS = 50000
CTRL_FREQ = 20
ACTION_SCALE = 1.2
CONTROL_DELTA_SCALE = 0.7

# Reachable targets around the arm base (world coordinates).
TARGETS = [
    [0.20, 0.00, 1.5],
]

# --- UPGRADED NETWORK ARCHITECTURE ---
# 3 (TCP) + 3 (Target) + 6 (Joint Angles) = 12 Inputs
INPUT_SIZE = 12 
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 6  # Lynx standalone has 6 joint actuators
GENOME_PATH = "best_lynx_brain_body.npy"
BRAIN_PATH = "best_lynx_brain_weights.npy"
MORPH_PATH = "best_lynx_morphology.npy"

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
# Home pose used to start each rollout in a consistent upright posture.
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


def evaluate_individual(genotype, target_position):
    """
    Evaluates a single individual's morphology and brain.
    Returns the minimum distance to the target achieved during the episode.
    """
    # 1. DECODE GENOTYPE
    tube_lengths = np.clip(genotype[0:5], 0.05, 0.4).tolist()
    rotations = np.clip(genotype[5:11], -np.pi, np.pi).tolist()
    nn_weights = genotype[11:] # Open-ended slice grabs all remaining weights

    # 2. CONSTRUCT MORPHOLOGY
    robot_desc = {
        "num_joints": 6,
        "genotype_tube": [1, 0, 0, 1, 0], # Matches your sim.yaml
        "genotype_joints": OUTPUT_SIZE,
        "tube_lengths": tube_lengths,
        "rotation_angles": rotations,
        "task": "reach"
    }
    
    world = TableWorld()
    robot = construct_lynx(robot_description_dict=robot_desc)
    world.spawn(robot.spec)
    
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Find IDs
    tcp_site_id = resolve_site_id(model, "end_effector_tcp")
    target_site_id = resolve_site_id(model, "target")
    if tcp_site_id == -1 or target_site_id == -1:
        return 999.0

    # Accept either a single target [x, y, z] or a target batch [[...], [...], ...].
    target_arr = np.asarray(target_position, dtype=np.float64)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(1, 3)
    if target_arr.ndim != 2 or target_arr.shape[1] != 3:
        return 999.0

    per_target_scores = []
    for tgt in target_arr:
        model.site_pos[target_site_id] = tgt
        apply_home_pose(model, data)
        mujoco.mj_forward(model, data)

        min_distance = float("inf")

        # 3. RUN SIMULATION
        for step in range(SIM_STEPS):
            if step % CTRL_FREQ == 0:
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

            current_dist = np.linalg.norm(data.site_xpos[tcp_site_id] - data.site_xpos[target_site_id])
            if current_dist < min_distance:
                min_distance = current_dist

        per_target_scores.append(float(min_distance))

    return float(np.mean(per_target_scores))

def create_run_database() -> tuple[sqlite3.Connection, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = DATA_DIR / f"fitness_{timestamp}.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_fitness (
            generation INTEGER PRIMARY KEY,
            best_fitness REAL NOT NULL,
            mean_fitness REAL NOT NULL,
            worst_fitness REAL NOT NULL,
            elapsed_sec REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn, db_path


def log_generation_fitness(
    conn: sqlite3.Connection,
    generation: int,
    best_fitness: float,
    mean_fitness: float,
    worst_fitness: float,
    elapsed_sec: float,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO generation_fitness(
            generation, best_fitness, mean_fitness, worst_fitness, elapsed_sec
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (generation, best_fitness, mean_fitness, worst_fitness, elapsed_sec),
    )
    conn.commit()


def plot_fitness_from_db(db_path: str | Path, show: bool = False) -> Path:
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT generation, best_fitness, mean_fitness, worst_fitness
        FROM generation_fitness
        ORDER BY generation ASC
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No fitness rows found in database: {db_path}")

    generations = [r[0] for r in rows]
    best_vals = [r[1] for r in rows]
    mean_vals = [r[2] for r in rows]
    worst_vals = [r[3] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_vals, label="Best", linewidth=2)
    plt.plot(generations, mean_vals, label="Mean", linewidth=1.5)
    plt.plot(generations, worst_vals, label="Worst", linewidth=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Lynx Evolution Fitness Over Generations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = db_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    return plot_path

def main():
    console.print(f"[bold green]Starting Lynx Co-Evolution (Brain + Body)[/bold green]")
    console.print(f"Total genome size: {TOTAL_GENOME_SIZE} parameters")

    db_conn, db_path = create_run_database()
    console.print(f"Logging fitness to: {db_path}")

    # Setup Nevergrad CMA-ES with a non-degenerate starting genome.
    init = np.zeros(TOTAL_GENOME_SIZE, dtype=np.float64)
    init[:5] = 0.2  # Start with usable tube lengths, not collapsed 0.05 tubes.
    init[11:] = np.random.normal(0.0, 0.10, size=NUM_WEIGHTS)
    parametrization = ng.p.Array(init=init)
    parametrization.set_mutation(sigma=0.20)
    
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=NUM_GENERATIONS * POP_SIZE, num_workers=POP_SIZE)
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for gen in range(NUM_GENERATIONS):
            start_time = time.time()
            
            # Ask for candidates
            candidates = [optimizer.ask() for _ in range(POP_SIZE)]
            genotypes = [c.value for c in candidates]
            
            # Evaluate in parallel
            fitnesses = list(executor.map(
                evaluate_individual, 
                genotypes, 
                [TARGETS] * POP_SIZE
            ))
            
            # Tell optimizer
            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)
                
            best_fit = min(fitnesses)
            mean_fit = float(np.mean(fitnesses))
            worst_fit = max(fitnesses)
            elapsed = time.time() - start_time
            log_generation_fitness(
                db_conn,
                generation=gen,
                best_fitness=float(best_fit),
                mean_fitness=mean_fit,
                worst_fitness=float(worst_fit),
                elapsed_sec=float(elapsed),
            )
            console.print(
                f"Gen {gen:03d} | Best: {best_fit:.4f} | Mean: {mean_fit:.4f} "
                f"| Worst: {worst_fit:.4f} | Time: {elapsed:.1f}s"
            )

    # Save best individual and split artifacts for explicit replay loading.
    best_genome = optimizer.provide_recommendation().value
    np.save(GENOME_PATH, best_genome)
    np.save(MORPH_PATH, best_genome[:11])
    np.save(BRAIN_PATH, best_genome[11:])
    db_conn.close()

    plot_path = plot_fitness_from_db(db_path)
    console.print(f"Saved genome to: {GENOME_PATH}")
    console.print(f"Saved morphology to: {MORPH_PATH}")
    console.print(f"Saved brain weights to: {BRAIN_PATH}")
    console.print(f"Fitness plot saved to: {plot_path}")
    console.print("[bold cyan]Evolution Complete![/bold cyan]")

if __name__ == "__main__":
    main()