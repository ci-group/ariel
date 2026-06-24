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


def _compute_u_hover(params, num_motors):
    k_w, k, w_min, w_max = params["k_w"], params["k"], params["w_min"], params["w_max"]
    W_hover = math.sqrt(GRAVITY / (k_w * num_motors))
    z = float(np.clip((W_hover - w_min) / (w_max - w_min), 0.0, 1.0))
    disc = (1.0 - k) ** 2 + 4.0 * k * z * z
    U_hover = (-(1.0 - k) + math.sqrt(max(disc, 0.0))) / (2.0 * k)
    return float(np.clip(2.0 * U_hover - 1.0, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Per-motor tilt mixer
#
# For a multirotor, roll/pitch correction works by *biasing thrust around
# the body* based on each motor's azimuthal position. Motor at azimuth φ
# contributes pitch torque ∝ cos(φ) and roll torque ∝ sin(φ). We compute
# this once from the propeller list — it's pure morphology, not learned.
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_mixer(propellers):
    """Return (N, 2) tensor: column 0 = pitch contribution, column 1 = roll.
    Each row is a per-motor gain for [pitch_err, roll_err] feedback.

    Sign convention (verified against src/ariel/simulation/drone/
    dynamics_params.py:118-120 and the torch_drone_gate_env body dynamics):
      * Roll:   k_p_signed[i] = -y_i · k_f / Ixx.  Right-side motor (+y) with
                phi>0 (right wing down in NED) → needs MORE thrust → mixer
                contribution is +sin(phi)·phi.  ✓ correct as written.
      * Pitch:  k_q_signed[i] = +x_i · k_f / Iyy.  Front motor (+x) with
                theta>0 (nose up in NED) → needs LESS thrust to push the nose
                back down → mixer contribution must be NEGATIVE.  Hence
                -cos(phi) here.  The asymmetric minus on y vs plus on x in
                dynamics_params is exactly why pitch and roll need opposite
                mixer signs even though both come from the same kinematic
                azimuth.
    """
    mix = np.zeros((len(propellers), 2), dtype=np.float32)
    for i, p in enumerate(propellers):
        pos = np.asarray(p["loc"], dtype=np.float32)  # body-frame position
        phi = math.atan2(float(pos[1]), float(pos[0]))
        mix[i, 0] = -math.cos(phi)   # pitch  (note negative — see docstring)
        mix[i, 1] =  math.sin(phi)   # roll
    return mix


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
        self.target = torch.tensor(HOVER_TARGET_NED, device=self.dev, dtype=self.dtype)
        self.u_hover = _compute_u_hover(params, n_mot)
        self.mix = torch.tensor(_tilt_mixer(propellers), device=self.dev, dtype=self.dtype)
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
        """Run one episode per param vector. Returns survival_steps (int)."""
        # Param layout per candidate:
        #   [trim(N), k_alt_p, k_alt_d, k_tilt, k_rate]
        N = self.n_mot
        trim    = params_batch[:, 0:N]                              # (λ, N)
        k_alt_p = params_batch[:, N + 0].unsqueeze(1)               # (λ, 1)
        k_alt_d = params_batch[:, N + 1].unsqueeze(1)               # (λ, 1)
        k_tilt  = params_batch[:, N + 2].unsqueeze(1)               # (λ, 1)
        k_rate  = params_batch[:, N + 3].unsqueeze(1)               # (λ, 1)

        s = self.reset()
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.dev)
        survival = torch.zeros(self.num_envs, dtype=self.dtype, device=self.dev)

        for _ in range(self.max_steps):
            z_err      = s[:, 2:3] - self.target[2]   # NED: z down, err > 0 → too low
            vz         = s[:, 5:6]                    # NED: vz > 0 → falling
            roll       = s[:, 6:7]                    # phi
            pitch      = s[:, 7:8]                    # theta
            roll_rate  = s[:, 9:10]                   # body p
            pitch_rate = s[:, 10:11]                  # body q

            # Feedback law (PD on altitude + PD on attitude via mixer):
            #   altitude:  +k_alt_p * z_err  - k_alt_d * vz
            #   attitude P: per-motor mixer * k_tilt * [pitch, roll]
            #   attitude D: per-motor mixer * k_rate * [pitch_rate, roll_rate]
            alt_cmd  = k_alt_p * z_err - k_alt_d * vz                                    # (λ, 1)
            att_cmd  = k_tilt * (self.mix[:, 0].unsqueeze(0) * pitch +                    # (λ, N)
                                 self.mix[:, 1].unsqueeze(0) * roll)
            rate_cmd = k_rate * (self.mix[:, 0].unsqueeze(0) * pitch_rate +               # (λ, N)
                                 self.mix[:, 1].unsqueeze(0) * roll_rate)
            action = trim + alt_cmd + att_cmd + rate_cmd                                  # (λ, N)
            action = (self.u_hover + action.clamp(-1.0, 1.0) * 0.4).clamp(-1.0, 1.0)

            sd = self._dyn(s.T, action.T).T
            s = s + self.dt * sd

            pos = s[:, 0:3]
            tilt = torch.norm(s[:, 6:8], dim=1)
            oob = (pos.abs() > 3.0).any(dim=1)
            flipped = tilt > (math.pi / 3)         # crashed if > 60° tilt
            divg = ~torch.isfinite(s).all(dim=1)
            dead = oob | flipped | divg

            # Dense fitness: smooth Gaussian on distance AND on attitude.
            #
            # Position-only reward let CMA collapse k_tilt and k_rate to zero
            # — the drone held altitude with no attitude correction and slowly
            # toppled out of the reward sphere. Adding an exp(-(tilt/σ_θ)²)
            # factor gives the attitude gains a real fitness gradient.
            #
            # Per Hansen 2023 §B.4: dense reward avoids the plateau that
            # caused the indicator version to stall.
            dist = torch.norm(pos - self.target.unsqueeze(0), dim=1)
            tilt = torch.norm(s[:, 6:8], dim=1)
            reward = torch.exp(-(dist / 0.4) ** 2) * torch.exp(-(tilt / 0.25) ** 2)
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

NUM_PARAMS = N + 4   # trim_i (N) + k_alt_p + k_alt_d + k_tilt + k_rate
console.log(f"Search space: {NUM_PARAMS} dims  (trim×{N} + alt_p + alt_d + tilt_p + rate_d)")

env = BatchedHover(
    propellers=propellers, num_envs=args.population,
    device=args.device, dt=DT, max_steps=args.episode_steps,
)

# CMA-ES.  Init: trims at 0 (u_hover is already analytically subtracted),
# feedback gains warm-started with the correct *signs* so we don't waste
# generations exploring the "no feedback" basin. These aren't tuned —
# CMA-ES will refine them — but they put us on the right side of zero.
init = np.zeros(NUM_PARAMS, dtype=np.float32)
init[N + 0] = 0.5    # k_alt_p — push up when below target
init[N + 1] = -0.5   # k_alt_d — damp downward velocity (NED: vz > 0 = falling)
init[N + 2] = 0.3    # k_tilt  — P term: corrective torque proportional to tilt angle
init[N + 3] = -0.15  # k_rate  — D term: damps angular velocity (opposite sign of k_tilt)

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
    f"k_tilt={best_vec[N+2]:+.3f}  k_rate={best_vec[N+3]:+.3f}"
)
console.log(f"Saved → {DATA}")
