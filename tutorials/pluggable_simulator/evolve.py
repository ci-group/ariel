"""Demonstrate the EA + PPO workflow on the Isaac Lab backend.

Outer evolutionary loop evolves drone arm lengths; inner PPO loop
trains a hover-to-goal policy for each candidate morphology. Both
improvements are visible from one run:

* **Within each individual:** ``rl_games`` logs PPO mean reward
  per epoch — that's the *learning* curve.
* **Across generations:** best / mean fitness over the population
  improves as the EA selects the morphologies that hover better —
  that's the *evolution* curve.

Implementation: one subprocess per individual. The parent holds the
EA state; each child runs a fresh Isaac Sim + rl_games training
process from ``train.py --blueprint-json ... --experiment-prefix
...``. Why subprocesses?  In-process reuse of ``DirectRLEnv`` across
genomes was unreliable in our tests (env teardown left the second
build hanging); subprocesses give each individual a clean Isaac Sim
state. Cost: ~6 s Isaac-Sim startup per individual.

Defaults are sized for a few-minute smoke (3 gen × 4 ind × 30
epochs); scale them up via CLI for a more convincing curve.

Run (from the ariel-isaaclab-train env with setup_conda_env.sh
already sourced — see tutorials/pluggable_simulator/README.md §3b):

    python tutorials/pluggable_simulator/evolve.py

Override defaults:

    python tutorials/pluggable_simulator/evolve.py \\
        --generations 5 --population 6 --epochs-per-eval 50
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = Path(__file__).with_name("train.py")
RUNS_DIR = REPO_ROOT / "runs"


def _log(msg: str) -> None:
    """stderr-flushed log lines (children's stdout/stderr stream through)."""
    sys.stderr.write(f"[evolve] {msg}\n")
    sys.stderr.flush()


# ---------- genome ---------------------------------------------------------------

@dataclass
class ArmLengthGenome:
    """Quad genome — just the four arm lengths.

    For the tutorial we keep the genotype dead-simple. Azimuths are
    fixed to the canonical X-config; motor orientation, propsize, and
    spin direction are constants. The EA's only knob is arm length;
    PPO is the other source of improvement. That makes the two
    dimensions (morphology vs. policy) visually distinct in the
    output.
    """

    arm_lengths: np.ndarray  # shape (n_arms,)

    @classmethod
    def default_quad(cls) -> "ArmLengthGenome":
        return cls(arm_lengths=np.full(4, 0.18))

    def mutate(self, sigma: float, rng: np.random.Generator) -> "ArmLengthGenome":
        new = self.arm_lengths + rng.normal(0.0, sigma, self.arm_lengths.shape)
        return ArmLengthGenome(arm_lengths=np.clip(new, 0.10, 0.30))

    def to_genome_matrix(self) -> np.ndarray:
        """Project to ``spherical_angular_to_blueprint``'s input format.

        Per-arm row: ``[mag, arm_az, arm_pitch, motor_az, motor_pitch, direction]``.
        Azimuths are evenly spaced around the body; everything else
        constant.
        """
        n = len(self.arm_lengths)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack(
            [
                self.arm_lengths,
                angles,
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.array([float(i % 2) for i in range(n)]),
            ],
            axis=1,
        )


# ---------- CLI -----------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--generations", type=int, default=3,
                   help="Number of EA generations. Default 3.")
    p.add_argument("--population", type=int, default=4,
                   help="Population size. Default 4.")
    p.add_argument("--epochs-per-eval", type=int, default=30,
                   help="rl_games PPO max_epochs per individual. Default 30.")
    p.add_argument("--num-envs", type=int, default=16,
                   help="Isaac Sim parallel envs per individual eval.")
    p.add_argument("--propsize", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-sigma", type=float, default=0.04,
                   help="Std of Gaussian noise around the default quad's "
                        "arm lengths when seeding the initial population.")
    p.add_argument("--mut-sigma", type=float, default=0.03,
                   help="Std of Gaussian noise applied to arm lengths after "
                        "tournament selection.")
    p.add_argument("--device-override", type=str, default=None,
                   help="Pass through to train.py (e.g., 'cpu' or 'cuda:0').")
    p.add_argument("--keep-blueprints", action="store_true",
                   help="Don't delete the per-individual blueprint JSON files.")
    return p


# ---------- main ----------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    _log("=== Evolution + PPO on Isaac Lab backend (subprocess mode) ===")
    _log(f"  generations       : {args.generations}")
    _log(f"  population        : {args.population}")
    _log(f"  epochs per eval   : {args.epochs_per_eval}")
    _log(f"  num_envs          : {args.num_envs}")
    _log(f"  seed              : {args.seed}")
    _log(f"  total evals       : {args.generations * args.population}")

    # Initial population: default quad + perturbations.
    population: list[ArmLengthGenome] = [ArmLengthGenome.default_quad()]
    for _ in range(args.population - 1):
        population.append(population[0].mutate(args.init_sigma, rng))

    history: list[dict] = []
    eval_counter = 0
    overall_start = time.time()

    with tempfile.TemporaryDirectory(prefix="ariel_evolve_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for gen in range(args.generations):
            _log("")
            _log(f"=== Generation {gen + 1}/{args.generations} ===")
            fitnesses: list[float] = []
            for ind_idx, individual in enumerate(population):
                eval_counter += 1
                exp_name = f"ariel_evolve_{eval_counter:04d}"
                bp_path = tmpdir_path / f"{exp_name}.json"

                # Render the blueprint and persist as JSON so the child
                # process loads exactly the same morphology.
                blueprint = spherical_angular_to_blueprint(
                    individual.to_genome_matrix(), propsize=args.propsize,
                )
                blueprint.save_json(bp_path)

                t0 = time.time()
                fitness = _evaluate_in_subprocess(
                    bp_path=bp_path,
                    exp_name=exp_name,
                    args=args,
                )
                dt = time.time() - t0
                fitnesses.append(fitness)
                _log(
                    f"  ind {ind_idx}: arm_lens="
                    f"{individual.arm_lengths.round(3).tolist()} "
                    f"fitness={fitness:.4f}  ({dt:.1f}s)"
                )

                if args.keep_blueprints:
                    keep_path = RUNS_DIR / f"{exp_name}.blueprint.json"
                    keep_path.parent.mkdir(parents=True, exist_ok=True)
                    keep_path.write_text(bp_path.read_text())

            best = max(fitnesses)
            worst = min(fitnesses)
            mean = sum(fitnesses) / len(fitnesses)
            history.append({"gen": gen, "best": best, "mean": mean, "worst": worst})
            _log(
                f"  -- gen {gen + 1}: best={best:.4f}  mean={mean:.4f}  "
                f"worst={worst:.4f}"
            )

            # Tournament-of-2 select → Gaussian mutate to fill new population.
            new_pop: list[ArmLengthGenome] = []
            for _ in range(len(population)):
                i1 = int(rng.integers(0, len(population)))
                i2 = int(rng.integers(0, len(population)))
                winner_idx = i1 if fitnesses[i1] >= fitnesses[i2] else i2
                new_pop.append(population[winner_idx].mutate(args.mut_sigma, rng))
            population = new_pop

    elapsed = time.time() - overall_start
    _log("")
    _log(f"=== Evolution complete (wall time {elapsed:.1f}s) ===")
    for h in history:
        _log(
            f"  gen {h['gen'] + 1}: best={h['best']:.4f}  "
            f"mean={h['mean']:.4f}  worst={h['worst']:.4f}"
        )


# ---------- per-individual subprocess --------------------------------------------

def _evaluate_in_subprocess(
    *, bp_path: Path, exp_name: str, args: argparse.Namespace,
) -> float:
    """Fork a child train.py for one individual; return its fitness."""
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--simulator", "isaaclab",
        "--mode", "train",
        "--headless",
        "--blueprint-json", str(bp_path),
        "--experiment-prefix", exp_name,
        "--max-iterations", str(args.epochs_per_eval),
        "--num-envs", str(args.num_envs),
        "--propsize", str(args.propsize),
        "--seed", str(args.seed),
    ]
    if args.device_override is not None:
        cmd += ["--device-override", args.device_override]

    # Stream the child's output through so the user can watch
    # PPO log lines accumulate per-individual.
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        _log(f"  WARN: child exited with code {result.returncode}; "
             f"recording fitness=nan")
        return float("nan")

    return _extract_reward_from_checkpoint(exp_name)


def _extract_reward_from_checkpoint(exp_name: str) -> float:
    """Parse the final episode reward out of rl_games' checkpoint filename.

    rl_games writes checkpoints to
    ``runs/<exp_name>_<timestamp>/nn/last_<exp_name>_ep_<E>_rew__<R>_.pth``.
    The reward in the filename is the value the runner last reported
    when it wrote that checkpoint — good enough for a fitness scalar.
    """
    matches = sorted(
        RUNS_DIR.glob(f"{exp_name}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        return float("nan")
    nn_dir = matches[-1] / "nn"
    if not nn_dir.exists():
        return float("nan")
    ckpts = list(nn_dir.glob("last_*.pth"))
    if not ckpts:
        return float("nan")
    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    m = re.search(r"rew_+(-?[\d.]+)_", latest.name)
    return float(m.group(1)) if m else float("nan")


if __name__ == "__main__":
    main()
