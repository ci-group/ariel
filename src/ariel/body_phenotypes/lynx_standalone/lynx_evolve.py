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
NUM_GENERATIONS = 100
POP_SIZE = 32
SIM_STEPS = 5000
CTRL_FREQ = 20
ACTION_SCALE = 0.8
CONTROL_DELTA_SCALE = 0.35

# CMA-ES tuning for smoother, less erratic learning.
CMA_INIT_SIGMA = 0.055
CMA_NUM_WORKERS = 10

# Precision-touch objective tuning (meters).
TOUCH_THRESHOLD = 0.006
HOLD_THRESHOLD = 0.015
APPROACH_THRESHOLD = 0.035

# Reachable targets around the arm base (world coordinates).
TARGETS = [
    [0.20, 0.00, 1.5]
]

# --- UPGRADED NETWORK ARCHITECTURE ---
# 3 (TCP) + 3 (Target) + 6 (Joint Angles) = 12 Inputs
INPUT_SIZE = 12 
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 6  # Lynx standalone has 6 joint actuators
GENOME_PATH = "best_lynx_brain_body.npy"
BRAIN_PATH = "best_lynx_brain_weights.npy"
MORPH_PATH = "best_lynx_morphology.npy"
MORPH_PARAM_SIZE = 11

# If False, keep morphology fixed and only evolve controller weights.
OPTIMIZE_MORPHOLOGY = False

# A known-reachable baseline morphology for controller-only training.
FIXED_TUBE_LENGTHS = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float64)
FIXED_ROTATIONS = np.array([0, 1.57, 0, 1.57, 0, 1.57])

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
TOTAL_GENOME_SIZE = int(MORPH_PARAM_SIZE + NUM_WEIGHTS) # 5 tubes + 6 rotations + weights
OPTIMIZED_GENOME_SIZE = int(TOTAL_GENOME_SIZE if OPTIMIZE_MORPHOLOGY else NUM_WEIGHTS)

def unflatten_weights(weights):
    """Safely unflattens the 1D genome into 3 layers ONCE per rollout."""
    assert len(weights) == NUM_WEIGHTS, f"Shape Mismatch! Weights is {len(weights)}, expected {NUM_WEIGHTS}"
    
    idx = 0
    W1 = weights[idx : idx + W1_SIZE].reshape(W1_SHAPE); idx += W1_SIZE
    b1 = weights[idx : idx + B1_SIZE].reshape(B1_SHAPE); idx += B1_SIZE
    
    W2 = weights[idx : idx + W2_SIZE].reshape(W2_SHAPE); idx += W2_SIZE
    b2 = weights[idx : idx + B2_SIZE].reshape(B2_SHAPE); idx += B2_SIZE
    
    W3 = weights[idx : idx + W3_SIZE].reshape(W3_SHAPE); idx += W3_SIZE
    b3 = weights[idx : idx + B3_SIZE].reshape(B3_SHAPE)
    
    return W1, b1, W2, b2, W3, b3

def fast_mlp(obs, W1, b1, W2, b2, W3, b3):
    """Fast forward-pass using pre-shaped matrices."""
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
    Returns the fitness score for the target (lower is better).
    """
    # 1. DECODE GENOTYPE
    geno = np.asarray(genotype, dtype=np.float64)
    if OPTIMIZE_MORPHOLOGY:
        if len(geno) != TOTAL_GENOME_SIZE:
            return 999.0
        morph = geno[:MORPH_PARAM_SIZE]
        nn_weights = geno[MORPH_PARAM_SIZE:]
    else:
        if len(geno) != NUM_WEIGHTS:
            return 999.0
        morph = np.concatenate([FIXED_TUBE_LENGTHS, FIXED_ROTATIONS])
        nn_weights = geno

    tube_lengths = np.clip(morph[0:5], 0.05, 0.4).tolist()
    rotations = np.clip(morph[5:MORPH_PARAM_SIZE], -np.pi, np.pi).tolist()

    # --- FIX 1: Unpack weights ONCE before the simulation loop ---
    W1, b1, W2, b2, W3, b3 = unflatten_weights(nn_weights)

    # 2. CONSTRUCT MORPHOLOGY
    robot_desc = {
        "num_joints": 6,
        "genotype_tube": [1, 1, 1, 1, 1], # Spacer tubes included!
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
        final_distance = float("inf")
        touch_steps = 0
        first_touch_step = SIM_STEPS
        first_approach_step = SIM_STEPS
        # We can still track cumulative distance for our own logs, but won't penalize it
        cumulative_distance = 0.0 

        # 3. RUN SIMULATION
        for step in range(SIM_STEPS):
            if step % CTRL_FREQ == 0:
                tcp_pos = data.site_xpos[tcp_site_id]
                target_p = data.site_xpos[target_site_id]
                joint_angles = get_joint_angles(model, data)
                rel_target = target_p - tcp_pos
                obs = np.concatenate([tcp_pos, rel_target, joint_angles])

                # --- FIX 2: Fast MLP execution ---
                raw_action = fast_mlp(obs, W1, b1, W2, b2, W3, b3)
                
                # --- FIX 3: Absolute Position Control ---
                # Network output [-1, 1] maps directly to joint target angles
                desired = raw_action * 2.8

                for i, jid in enumerate(get_actuated_joint_ids(model, count=OUTPUT_SIZE)):
                    if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
                        lo, hi = model.jnt_range[jid]
                        desired[i] = np.clip(desired[i], lo, hi)

                data.ctrl[:OUTPUT_SIZE] = desired

            mujoco.mj_step(model, data)

            current_dist = np.linalg.norm(data.site_xpos[tcp_site_id] - data.site_xpos[target_site_id])
            cumulative_distance += float(current_dist)
            
            if current_dist < min_distance:
                min_distance = current_dist
            if current_dist <= HOLD_THRESHOLD:
                touch_steps += 1
            if current_dist <= APPROACH_THRESHOLD and first_approach_step == SIM_STEPS:
                first_approach_step = step
            if current_dist <= TOUCH_THRESHOLD and first_touch_step == SIM_STEPS:
                first_touch_step = step

        final_distance = float(
            np.linalg.norm(data.site_xpos[tcp_site_id] - data.site_xpos[target_site_id])
        )

        touched = first_touch_step < SIM_STEPS
        touch_latency = (first_touch_step / SIM_STEPS) if touched else 1.0
        approached = first_approach_step < SIM_STEPS
        approach_latency = (first_approach_step / SIM_STEPS) if approached else 1.0
        hold_ratio = touch_steps / SIM_STEPS

        # --- FIX 4: Updated Fitness ---
        # Removed the mean distance penalty so the robot isn't afraid to move!
        target_score = (
            0.60 * float(min_distance)
            + 0.40 * final_distance
            + 0.16 * touch_latency
            + 0.10 * approach_latency
            - 0.12 * hold_ratio
            + (0.10 if not approached else 0.0)
            + (0.08 if not touched else 0.0)
        )

        per_target_scores.append(float(target_score))

    return float(np.mean(per_target_scores))

def create_run_database() -> tuple[sqlite3.Connection, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = DATA_DIR / f"fitness_{timestamp}.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generations (
            generation INTEGER PRIMARY KEY,
            best_fitness REAL NOT NULL,
            mean_fitness REAL NOT NULL,
            std_fitness REAL NOT NULL,
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
    std_fitness: float,
    worst_fitness: float,
    elapsed_sec: float,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO generations(
            generation, best_fitness, mean_fitness, std_fitness, worst_fitness, elapsed_sec
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (generation, best_fitness, mean_fitness, std_fitness, worst_fitness, elapsed_sec),
    )
    conn.commit()


def plot_fitness_from_db(db_path: str | Path, show: bool = False) -> Path:
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT generation, mean_fitness, std_fitness
        FROM generations
        ORDER BY generation ASC
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No fitness rows found in database: {db_path}")

    generations = [r[0] for r in rows]
    mean_vals = np.array([r[1] for r in rows], dtype=np.float64)
    std_vals = np.array([r[2] for r in rows], dtype=np.float64)
    lower = mean_vals - std_vals
    upper = mean_vals + std_vals

    plt.figure(figsize=(10, 5))
    plt.plot(generations, mean_vals, label="Mean", linewidth=2.0)
    plt.fill_between(generations, lower, upper, alpha=0.25, label="Mean ± 1 std")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Lynx Evolution Fitness (Mean and Std) Over Generations")
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
    # 1. SETUP DATABASE
    db_path = DATA_DIR / f"evolution_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    db_conn = sqlite3.connect(db_path)
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            generation INTEGER PRIMARY KEY,
            best_fitness REAL,
            mean_fitness REAL,
            std_fitness REAL,
            worst_fitness REAL,
            elapsed_sec REAL
        )
    """)
    db_conn.commit()

    # --- FIX 5: Zero-Initialization for Neural Network Weights ---
    # Setup Nevergrad CMA-ES with a non-degenerate starting genome.
    init = np.zeros(OPTIMIZED_GENOME_SIZE, dtype=np.float64)
    if OPTIMIZE_MORPHOLOGY:
        init[:5] = FIXED_TUBE_LENGTHS  # Start with usable tube lengths.
        init[MORPH_PARAM_SIZE:] = 0.0  # Start weights exactly at zero
    else:
        init[:] = 0.0                  # Start weights exactly at zero
        
    parametrization = ng.p.Array(init=init)
    parametrization.set_mutation(sigma=CMA_INIT_SIGMA)
    
    optimizer = ng.optimizers.CMA(
        parametrization=parametrization, 
        budget=NUM_GENERATIONS * POP_SIZE,
        num_workers=CMA_NUM_WORKERS
    )

    console.print(f"\n[bold green]Starting Evolution for {NUM_GENERATIONS} generations...[/bold green]")
    console.print(f"Tracking targets: {TARGETS}\n")

    # 2. EVOLUTION LOOP
    with ProcessPoolExecutor(max_workers=CMA_NUM_WORKERS) as executor:
        for gen in range(1, NUM_GENERATIONS + 1):
            start_time = time.time()
            
            # Ask for candidates
            candidates = [optimizer.ask() for _ in range(POP_SIZE)]
            genomes = [c.value for c in candidates]
            
            # Evaluate in parallel
            futures = [executor.submit(evaluate_individual, geno, TARGETS) for geno in genomes]
            fitnesses = [f.result() for f in futures]
            
            # Tell the optimizer the results
            for c, fit in zip(candidates, fitnesses):
                optimizer.tell(c, fit)
            
            # Calculate metrics
            best_fit = np.min(fitnesses)
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            worst_fit = np.max(fitnesses)
            elapsed = time.time() - start_time
            
            # Log to DB
            db_conn.execute(
                """
                INSERT INTO generations 
                (generation, best_fitness, mean_fitness, std_fitness, worst_fitness, elapsed_sec)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    gen,
                    float(best_fit),
                    float(mean_fit),
                    float(std_fit),
                    float(worst_fit),
                    float(elapsed),
                )
            )
            db_conn.commit()

            # Console output
            console.print(
                f"Gen {gen:03d} | Best: {best_fit:.4f} | Mean: {mean_fit:.4f} "
                f"| Worst: {worst_fit:.4f} | Time: {elapsed:.1f}s"
            )

    # 3. SAVE BEST ARTIFACTS
    # Combined genome is always full-size for replay compatibility.
    best_encoded = optimizer.provide_recommendation().value
    if OPTIMIZE_MORPHOLOGY:
        best_morph = np.asarray(best_encoded[:MORPH_PARAM_SIZE], dtype=np.float64)
        best_brain = np.asarray(best_encoded[MORPH_PARAM_SIZE:], dtype=np.float64)
    else:
        best_morph = np.concatenate([FIXED_TUBE_LENGTHS, FIXED_ROTATIONS]).astype(np.float64)
        best_brain = np.asarray(best_encoded, dtype=np.float64)

    best_genome = np.concatenate([best_morph, best_brain])
    np.save(GENOME_PATH, best_genome)
    np.save(MORPH_PATH, best_morph)
    np.save(BRAIN_PATH, best_brain)
    db_conn.close()

    try:
        plot_path = plot_fitness_from_db(db_path)
        console.print(f"\n[bold blue]Saved fitness plot to {plot_path}[/bold blue]")
    except NameError:
        pass # plot_fitness_from_db might not be defined depending on your imports, safely ignore
        
    console.print("[bold green]Evolution Complete! You can now run replay_best.py[/bold green]")

if __name__ == "__main__":
    main()
