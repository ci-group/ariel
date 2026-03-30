import argparse
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.lynx_mjspec.lynx_arm import LynxArm
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld

console = Console()

DATA_DIR = Path.cwd() / "__data__" / "lynx_mjspec"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_JOINTS = 6
NUM_TUBES = 5

TUBE_MIN = 0.1
TUBE_MAX = 1.0

DEFAULT_TARGET = [0.20, 0.00, 1.20]
DEFAULT_SIM_STEPS = 3500
DEFAULT_CTRL_FREQ = 20

DEFAULT_POP_SIZE = 32
DEFAULT_GENERATIONS = 100
DEFAULT_SIGMA = 0.10
DEFAULT_WORKERS = max(1, (Path('/proc/cpuinfo').read_text().count('processor\t:') if Path('/proc/cpuinfo').exists() else 8) // 2)

TOUCH_THRESHOLD = 0.01
HOLD_THRESHOLD = 0.02

INVALID_FITNESS = 999.0

GENOME_PATH = "best_lynx_mjspec_controller_tubes_genome.npy"
TUBES_PATH = "best_lynx_mjspec_tube_lengths.npy"
BRAIN_PATH = "best_lynx_mjspec_brain_weights.npy"


class FastNumpyNetwork:
    """Fast 3-layer NumPy MLP similar to the RE book example."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        w1_end = hidden_size * input_size
        b1_end = w1_end + hidden_size

        w2_end = b1_end + (hidden_size * hidden_size)
        b2_end = w2_end + hidden_size

        w3_end = b2_end + (output_size * hidden_size)
        b3_end = w3_end + output_size

        if len(weights) != b3_end:
            raise ValueError(f"Invalid weight size {len(weights)}, expected {b3_end}")

        self.w1 = weights[0:w1_end].reshape(hidden_size, input_size)
        self.b1 = weights[w1_end:b1_end]

        self.w2 = weights[b1_end:w2_end].reshape(hidden_size, hidden_size)
        self.b2 = weights[w2_end:b2_end]

        self.w3 = weights[b2_end:w3_end].reshape(output_size, hidden_size)
        self.b3 = weights[w3_end:b3_end]

    @staticmethod
    def _elu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, np.exp(x) - 1.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.dot(self.w1, x) + self.b1
        x = self._elu(x)

        x = np.dot(self.w2, x) + self.b2
        x = self._elu(x)

        x = np.dot(self.w3, x) + self.b3
        return np.tanh(x)


def get_actuated_joint_ids(model: mujoco.MjModel, count: int = NUM_JOINTS) -> list[int]:
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


def get_joint_state(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros(NUM_JOINTS, dtype=np.float64)
    qd = np.zeros(NUM_JOINTS, dtype=np.float64)
    for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
        qaddr = int(model.jnt_qposadr[jid])
        daddr = int(model.jnt_dofadr[jid])
        q[i] = data.qpos[qaddr]
        qd[i] = data.qvel[daddr]
    return q, qd


def build_model(tube_lengths: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData, int, int, list[int]]:
    config = {
        "num_joints": 6,
        "genotype_tube": [1, 1, 1, 1, 1],
        "genotype_joints": 6,
        "tube_lengths": np.clip(tube_lengths, TUBE_MIN, TUBE_MAX).tolist(),
        "rotation_angles": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
        "task": "reach",
    }

    arm = LynxArm(config=config)
    world = TableWorld()
    world.spawn(arm.spec)

    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)

    tcp_site_id = resolve_site_id(model, "tcp")
    target_site_id = resolve_site_id(model, "target")
    if tcp_site_id == -1 or target_site_id == -1:
        raise RuntimeError("Could not resolve TCP/target site IDs")

    joint_ids = get_actuated_joint_ids(model, count=NUM_JOINTS)
    return model, data, tcp_site_id, target_site_id, joint_ids


def evaluate_candidate(
    genotype: np.ndarray,
    input_size: int,
    hidden_size: int,
    output_size: int,
    target: np.ndarray,
    sim_steps: int,
    ctrl_freq: int,
) -> float:
    try:
        tube_lengths = genotype[:NUM_TUBES]
        weights = genotype[NUM_TUBES:]

        model, data, tcp_sid, tgt_sid, joint_ids = build_model(tube_lengths)
        model.site_pos[tgt_sid] = target
        mujoco.mj_forward(model, data)

        net = FastNumpyNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights=weights)

        min_distance = float("inf")
        final_distance = float("inf")
        touch_steps = 0
        first_touch_step = sim_steps

        action = np.zeros(NUM_JOINTS, dtype=np.float64)

        for step in range(sim_steps):
            if step % ctrl_freq == 0:
                tcp_pos = data.site_xpos[tcp_sid]
                target_pos = data.site_xpos[tgt_sid]
                rel_target = target_pos - tcp_pos

                q, qd = get_joint_state(model, data, joint_ids)
                phase = np.array([
                    2.0 * np.sin(data.time * 2.0 * np.pi),
                    2.0 * np.cos(data.time * 2.0 * np.pi),
                ])

                obs = np.concatenate([q, qd, rel_target, phase]).astype(np.float64)
                action = net.forward(obs) * 0.35
                desired = q + action

                for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
                    if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
                        lo, hi = model.jnt_range[jid]
                        desired[i] = np.clip(desired[i], lo, hi)

                data.ctrl[:NUM_JOINTS] = desired

            mujoco.mj_step(model, data)
            d = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
            final_distance = d
            if d < min_distance:
                min_distance = d
            if d <= HOLD_THRESHOLD:
                touch_steps += 1
            if d <= TOUCH_THRESHOLD and first_touch_step == sim_steps:
                first_touch_step = step

        touched = first_touch_step < sim_steps
        touch_latency = (first_touch_step / sim_steps) if touched else 1.0
        hold_ratio = touch_steps / sim_steps

        return (
            0.60 * min_distance
            + 0.40 * final_distance
            + 0.12 * touch_latency
            - 0.10 * hold_ratio
            + (0.08 if not touched else 0.0)
        )
    except Exception:
        return INVALID_FITNESS


def create_run_database() -> tuple[sqlite3.Connection, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = DATA_DIR / f"lynx_mjspec_evo_{timestamp}.db"
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


def log_generation(
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


def plot_fitness_from_db(db_path: str | Path) -> Path:
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

    plt.figure(figsize=(10, 5))
    plt.plot(generations, mean_vals, label="Mean", linewidth=2.0)
    plt.fill_between(generations, mean_vals - std_vals, mean_vals + std_vals, alpha=0.25, label="Mean ± 1 std")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Lynx MjSpec Evolution (Controller + Tube Lengths)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = db_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve Lynx MjSpec controller and tube lengths with CMA-ES")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--population", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--ctrl-freq", type=int, default=DEFAULT_CTRL_FREQ)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET[0])
    parser.add_argument("--target-y", type=float, default=DEFAULT_TARGET[1])
    parser.add_argument("--target-z", type=float, default=DEFAULT_TARGET[2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
    input_size = NUM_JOINTS + NUM_JOINTS + 3 + 2
    hidden_size = int(args.hidden_size)
    output_size = NUM_JOINTS

    layer1_size = (hidden_size * input_size) + hidden_size
    layer2_size = (hidden_size * hidden_size) + hidden_size
    layer3_size = (output_size * hidden_size) + output_size
    num_weights = layer1_size + layer2_size + layer3_size

    genome_size = NUM_TUBES + num_weights

    # Tube lengths + MLP weights
    init = np.zeros(genome_size, dtype=np.float64)
    init[:NUM_TUBES] = 0.10
    init[NUM_TUBES:] = np.random.uniform(-0.1, 0.1, size=num_weights)

    parametrization = ng.p.Array(init=init)
    parametrization.set_mutation(sigma=float(args.sigma))

    optimizer = ng.optimizers.CMA(
        parametrization=parametrization,
        budget=int(args.generations) * int(args.population),
        num_workers=min(int(args.population), int(args.workers)),
    )

    db_conn, db_path = create_run_database()

    console.print("[bold green]Starting Lynx MjSpec evolution (controller + tube lengths)[/bold green]")
    console.print(f"Target: {target.tolist()}")
    console.print(f"Genome size: {genome_size} ({NUM_TUBES} tube genes + {num_weights} network weights)")

    with ProcessPoolExecutor(max_workers=min(int(args.population), int(args.workers))) as executor:
        for gen in range(1, int(args.generations) + 1):
            start = time.time()

            candidates = [optimizer.ask() for _ in range(int(args.population))]
            genomes = [np.asarray(c.value, dtype=np.float64) for c in candidates]

            fitnesses = list(
                executor.map(
                    evaluate_candidate,
                    genomes,
                    [input_size] * len(genomes),
                    [hidden_size] * len(genomes),
                    [output_size] * len(genomes),
                    [target] * len(genomes),
                    [int(args.sim_steps)] * len(genomes),
                    [int(args.ctrl_freq)] * len(genomes),
                )
            )

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, float(fit))

            best_fit = float(np.min(fitnesses))
            mean_fit = float(np.mean(fitnesses))
            std_fit = float(np.std(fitnesses))
            worst_fit = float(np.max(fitnesses))
            elapsed = float(time.time() - start)

            log_generation(
                db_conn,
                generation=gen,
                best_fitness=best_fit,
                mean_fitness=mean_fit,
                std_fitness=std_fit,
                worst_fitness=worst_fit,
                elapsed_sec=elapsed,
            )

            console.print(
                f"Gen {gen:03d} | Best: {best_fit:.4f} | Mean: {mean_fit:.4f} "
                f"| Worst: {worst_fit:.4f} | Time: {elapsed:.1f}s"
            )

    best_genome = np.asarray(optimizer.provide_recommendation().value, dtype=np.float64)
    best_tube_lengths = np.clip(best_genome[:NUM_TUBES], TUBE_MIN, TUBE_MAX)
    best_brain = best_genome[NUM_TUBES:]

    np.save(GENOME_PATH, best_genome)
    np.save(TUBES_PATH, best_tube_lengths)
    np.save(BRAIN_PATH, best_brain)
    db_conn.close()

    plot_path = plot_fitness_from_db(db_path)

    console.print(f"Saved best genome to: {GENOME_PATH}")
    console.print(f"Saved best tube lengths to: {TUBES_PATH}")
    console.print(f"Saved best brain weights to: {BRAIN_PATH}")
    console.print(f"Saved run DB to: {db_path}")
    console.print(f"Saved fitness plot to: {plot_path}")
    
if __name__ == "__main__":
    main()
