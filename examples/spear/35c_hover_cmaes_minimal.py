"""Minimal CMA-ES hover: find ~12 numbers, not a neural net.

Idea: for a fixed-morphology hex, "hover" is essentially trim
calibration + a small linear feedback law. CMA-ES on a tiny search space
with a sparse fitness (= how long it survives) avoids every problem the
NN version of 35_hover_cmaes_nn.py has:

  * No σ-collapse onto a flat reward basin — the fitness IS sparse, so
    every improvement is a step-function ("survived 1 more second").
  * No obs scaling, no reward shaping, no reset-noise tuning. Crash =
    bad, survive = good.
  * Param count = N_motors (trim) + 3 (altitude P, D, tilt P)
    = 9 for a hex. CMA-ES converges on n=9 in 50–100 generations easily.

The controller is:

    motor_i = u_hover + trim_i + k_alt_p * z_err + k_alt_d * vz + k_tilt * tilt_i

where tilt_i is a hand-derived per-motor mixer from roll/pitch error
(see ``_tilt_mix`` below — same structure as a classic PID quad).

Run:
    uv run examples/spear/35c_hover_cmaes_minimal.py --blueprint <bp.json>
"""

import argparse
import math
import sys
import time
from pathlib import Path

import nevergrad as ng
import numpy as np
import torch
from rich.console import Console

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.dynamics_params import derive_reference_params
from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics

# Canonical hover prior: mixers, u_hover, controller formula.
# Single source of truth shared by 35c (this file), 35d (replay), and the
# Stage-2 residual env. Sign-convention regression is guarded by
# examples/spear/library/test_prior_controller.py.
sys.path.insert(0, str(Path(__file__).parent / "library"))
from prior_controller import HoverPrior, N_GAINS  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(description="Minimal CMA-ES hover (trim + linear feedback)")
parser.add_argument("--blueprint", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--budget", type=int, default=100, help="CMA-ES generations") # 200
# Default λ for n≈10: pycma uses 4 + 3·ln(n) ≈ 10; we use 24 — enough headroom
# for noisy fitness (random reset) without burning compute. Per CMA-ES tutorial
# (Hansen 2023, §Population Size): "increasing λ improves global search and
# robustness, at the price of slower convergence per function evaluation."
parser.add_argument("--population", type=int, default=128, help="λ")
parser.add_argument("--episode-steps", type=int, default=1000, help="Max sim steps (6s @ dt=0.01)")
parser.add_argument("--out-dir", default=f"__data__/hover_cmaes_min/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Constants + dynamics setup (same as 35_hover_cmaes_nn.py)
# ─────────────────────────────────────────────────────────────────────────────

HOVER_TARGET_NED = np.array([0.0, 0.0, -1.5], dtype=np.float32)
GRAVITY = 9.81
DT = 0.01


def _build_params(propellers):
    cfg = DroneConfiguration(propellers)
    params = derive_reference_params(
        propellers=cfg.propellers, mass=float(cfg.mass),
        inertia=np.asarray(cfg.inertia_matrix),
        prop_size=propellers[0].get("propsize", 2),
        gravity=GRAVITY,
    )
    return params, cfg.num_motors


# Mixers, u_hover, and controller formula now live in
# examples/spear/library/prior_controller.py (imported above as HoverPrior).
# Sign-convention tests live in test_prior_controller.py.


# ─────────────────────────────────────────────────────────────────────────────
# Batched rollout — λ envs, all evaluated in one tensor op
# ─────────────────────────────────────────────────────────────────────────────

class BatchedHover:
    def __init__(self, propellers, num_envs, device, dt, max_steps):
        params, n_mot = _build_params(propellers)
        self.dev = torch.device(device)
        self.dtype = torch.float32
        self.n_mot = n_mot
        self.dt = dt
        self.max_steps = max_steps
        self._dyn = _build_torch_dynamics(params, n_mot, GRAVITY, self.dev, self.dtype)
        # Single source of truth for the analytical controller.
        self.prior = HoverPrior(
            propellers=propellers, params=params,
            target_ned=HOVER_TARGET_NED.tolist(),
            gravity=GRAVITY, action_scale=0.4,
            device=self.dev, dtype=self.dtype,
        )
        # Aliases used elsewhere in this file (reset, reward computation).
        self.target = self.prior.target
        self.num_envs = num_envs
        self.state_dim = 12 + n_mot

    def reset(self):
        """Reset all envs to a slightly perturbed hover state. CMA-ES needs
        *some* variation to discriminate candidates, but we keep it small
        so the sparse fitness signal isn't dominated by init noise."""
        s = torch.zeros((self.num_envs, self.state_dim), device=self.dev, dtype=self.dtype)
        s[:, 0:3] = self.target.unsqueeze(0)
        s[:, 0:3] += (torch.rand((self.num_envs, 3), device=self.dev, dtype=self.dtype) - 0.5) * 0.2
        s[:, 6:8] = (torch.rand((self.num_envs, 2), device=self.dev, dtype=self.dtype) - 0.5) * 0.1  # tiny tilt
        return s

    @torch.no_grad()
    def rollout(self, params_batch):
        """Run one episode per param vector. Returns survival_steps (int).

        Controller logic lives in `prior.prior_action`. Param layout
        documented in `prior_controller.HoverPrior`:
        ``[trim×N, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate]``.
        """
        s = self.reset()
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.dev)
        survival = torch.zeros(self.num_envs, dtype=self.dtype, device=self.dev)

        for _ in range(self.max_steps):
            action = self.prior.prior_action(s, params_batch)
            sd = self._dyn(s.T, action.T).T
            s = s + self.dt * sd
            yaw_rate = s[:, 11:12]   # used below for reward shaping

            pos = s[:, 0:3]
            tilt = torch.norm(s[:, 6:8], dim=1)
            oob = (pos.abs() > 3.0).any(dim=1)
            flipped = tilt > (math.pi / 3)         # crashed if > 60° tilt
            divg = ~torch.isfinite(s).all(dim=1)
            dead = oob | flipped | divg

            # Dense fitness: smooth Gaussian on distance AND on attitude
            # AND on body yaw rate.
            #
            # Yaw-rate term added because spin imbalance (or just initial
            # perturbation) causes drones to spin up indefinitely without
            # feedback — pitch/roll stay nice but yaw runs away.  Without
            # this penalty, CMA leaves k_yaw_rate at 0 since the position
            # and tilt terms don't see yaw spin.  σ_r = 1.0 rad/s (~57°/s):
            # tight enough to punish noticeable spin, loose enough that
            # small recovery transients don't crush reward.
            dist = torch.norm(pos - self.target.unsqueeze(0), dim=1)
            tilt = torch.norm(s[:, 6:8], dim=1)
            r_abs = yaw_rate.abs().squeeze(-1)
            reward = (torch.exp(-(dist / 0.4) ** 2)
                      * torch.exp(-(tilt / 0.25) ** 2)
                      * torch.exp(-(r_abs / 1.0) ** 2))
            survival = torch.where(alive & ~dead,
                                   survival + reward, survival)
            alive = alive & ~dead
            if not alive.any():
                break

        return survival.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

bp = DroneBlueprint.load_json(Path(args.blueprint))
propellers = blueprint_to_propellers(bp, convention="ned")
N = len(propellers)
console.log(f"Motors: {N}")

env = BatchedHover(
    propellers=propellers, num_envs=args.population,
    device=args.device, dt=DT, max_steps=args.episode_steps,
)
NUM_PARAMS = env.prior.param_dim   # N + 5
console.log(f"Search space: {NUM_PARAMS} dims  (trim×{N} + alt_p + alt_d + tilt_p + rate_d + yaw_d)")

# CMA-ES.  Trims start at 0 (u_hover is already analytically subtracted),
# feedback gains warm-started with the correct *signs* via
# prior.default_init_params(). Single source of truth for the warm-start
# lives in prior_controller.py.
init = env.prior.default_init_params()

# σ₀ rule (Hansen 2023, Table 1): σ₀ = 0.3 · (b − a). Search range ≈ ±2 → 0.3·4 = 1.2
# is too aggressive given the warm-started signs. We use 0.4 — big enough that
# 3σ covers ±1.2 around each init coordinate (which contains the relevant basin)
# without making CMA overshoot the warm-start in the first few generations.
param = ng.p.Array(init=init)
param.set_mutation(sigma=0.4)
# NOTE: bounds removed at the parametrization level. Naive clipping breaks
# CMA's distributional assumptions (Hansen 2023 §B.5). Instead we let pycma's
# BoundPenalty handler (set via inopts below) penalize out-of-box samples.

# ParametrizedCMA with pycma `inopts` passthrough — see ParametrizedCMA.md wiki
# entry. Key choices:
#   * tolflatfitness=50 — patience on plateaus; with the dense Gaussian fitness
#     this should rarely trigger, but it's the documented escape hatch.
#   * CMA_active=True   — enable active CMA negative weights (recommended for
#     small populations per pycma defaults).
#   * CMA_diagonal=20   — first 20 iters use diagonal-only C (cheap, no rotation
#     learned). For n=10 with roughly axis-aligned scaling this saves compute
#     without losing convergence quality.
#   * bounds=[-2.0, 2.0] — pycma's default BoundaryHandler is BoundPenalty,
#     which applies a quadratic penalty on box violations (Hansen 2023 §B.5
#     recommended scheme). No need to name the handler explicitly.
optimizer = ng.optimizers.ParametrizedCMA(
    popsize=args.population,
    elitist=False,
    diagonal=False,
    inopts={
        "tolflatfitness": 50,
        "CMA_active": True,
        "CMA_diagonal": 20,
        "bounds": [-2.0, 2.0],
        "seed": int(args.seed),
        "verbose": -9,
    },
)(
    parametrization=param,
    budget=args.budget * args.population,
    num_workers=args.population,
)

console.rule(f"[bold blue]Minimal CMA-ES hover  λ={args.population}  budget={args.budget}")
t0 = time.time()
best_survival = 0.0
best_vec = init.copy()
# Per-generation stats. Each row appended once per generation. Saved to
# fitness.txt at end so 35d_replay_cmaes_minimal.py can plot it.
# Columns: gen, max, median, mean, std, min, best_ever
fitness_log = []

for gen in range(args.budget):
    candidates = [optimizer.ask() for _ in range(args.population)]
    P = np.stack([c.value for c in candidates], axis=0).astype(np.float32)
    survival = env.rollout(torch.as_tensor(P, device=env.dev, dtype=env.dtype))
    # CMA-ES minimises, so negate survival.
    losses = (-survival).astype(np.float32)

    for c, l in zip(candidates, losses):
        optimizer.tell(c, float(l))

    gen_max  = float(survival.max())
    gen_med  = float(np.median(survival))
    gen_mean = float(survival.mean())
    gen_std  = float(survival.std())
    gen_min  = float(survival.min())
    if gen_max > best_survival:
        best_survival = gen_max
        best_vec = P[int(np.argmax(survival))]
    fitness_log.append((gen + 1, gen_max, gen_med, gen_mean, gen_std, gen_min, best_survival))

    console.log(
        f"gen {gen+1:>3}/{args.budget}  "
        f"reward max={gen_max:>6.1f}  median={gen_med:>6.1f}  "
        f"best-ever={best_survival:>6.1f} / {args.episode_steps}  "
        f"({time.time()-t0:.1f}s)"
    )
    if gen_med >= args.episode_steps - 5:
        console.log("[green]Median candidate survives full episode — converged.[/green]")
        break

DATA = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
np.save(DATA / "best_params.npy", best_vec)
bp.save_json(DATA / "blueprint.json")

# Persist per-generation fitness. Tab-separated so it parses cleanly with
# np.loadtxt; header documents columns so the file is self-describing.
fit_arr = np.asarray(fitness_log, dtype=np.float32)
np.savetxt(
    DATA / "fitness.txt", fit_arr,
    header="gen\tmax\tmedian\tmean\tstd\tmin\tbest_ever",
    fmt=["%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f"],
    delimiter="\t",
)
trim_str = ", ".join(f"{x:+.3f}" for x in best_vec[:N])
console.log(
    f"\nBest cumulative reward {best_survival:.1f} / {args.episode_steps} "
    f"(≈ {best_survival*DT:.2f}s near target)\n"
    f"  trim   = [{trim_str}]\n"
    f"  k_alt_p={best_vec[N]:+.3f}  k_alt_d={best_vec[N+1]:+.3f}  "
    f"k_tilt={best_vec[N+2]:+.3f}  k_rate={best_vec[N+3]:+.3f}  "
    f"k_yaw_rate={best_vec[N+4]:+.3f}"
)
console.log(f"Saved → {DATA}")
