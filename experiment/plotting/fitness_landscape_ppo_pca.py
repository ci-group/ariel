"""PCA fitness landscape for the PPO undirected-locomotion policy.

Companion to ``fitness_landscape_cmaes_pca.py`` for the RL side of the
experiment. Reads a ``params_snapshots.pkl`` (one params tuple captured at
each eval boundary during training), fits a 2-component PCA across the
flattened policy_params trajectory, and evaluates a grid of perturbed
policies around the final params by running MJX rollouts on the GPU.

Loads
-----
    <run-dir>/ppo_config.json          — network factory kwargs + normalize flag
    <run-dir>/params.pkl               — final (normalizer, policy, value) tuple
    <run-dir>/params_snapshots.pkl     — list[(step, params_tuple)]
    <run-dir>/training_history.json    — per-eval reward curve

Writes
------
    <run-dir>/landscape_ppo_<basis>_<grid>x<grid>.png
    <run-dir>/landscape_ppo_<basis>_reward_<grid>x<grid>.png (if --reward)
    <run-dir>/landscape_ppo_<basis>_std_<grid>x<grid>.png    (if --seeds-per-cell > 1)
    <run-dir>/landscape_ppo_<basis>_1d_<grid>x<grid>.png
    <run-dir>/landscape_ppo_scree.png
    <run-dir>/landscape_ppo_reward_curve.png
    <run-dir>/landscape_ppo_<basis>_wave_<grid>x<grid>.png
    <run-dir>/landscape_ppo_<basis>_wave_picks_<grid>x<grid>.png
    <run-dir>/landscape_ppo_<basis>_migration_<grid>x<grid>.gif  (if --animate)

Notes
-----
    * ``--basis`` chooses PC directions (trajectory-aligned) or random directions.
      With ``both`` we emit the full set for each basis.
    * ``--filter-normalize`` (on by default) applies per-column kernel
      normalization (Li et al. 2018) so that one unit of α moves each layer by
      a consistent fraction of that layer's weight norm. This makes landscape
      shapes comparable across runs/morphologies.
    * ``--reward`` additionally plots the mean-reward surface alongside
      displacement; the training reward is the actual quantity PPO optimises.
    * ``--seeds-per-cell`` averages across N reset seeds; a separate std
      heatmap quantifies noise.

Example
-------
    /home/user/Desktop/EvoDevo/mujoco_playground/.venv/bin/python \
        experiment/plotting/fitness_landscape_ppo_pca.py \
        --run-dir __data__/undirected_locomotion_ppo_jax/insect_undirected-<timestamp>
"""

from __future__ import annotations

# XLA/MuJoCo configuration — must happen before jax and brax import.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

# Sibling import from the training script's location (experiment/rl/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
from undirected_locomotion_ppo_jax import (  # noqa: E402
    NACCDMAX_PER_ENV,
    NACONMAX_PER_ENV,
    UndirectedLocomotionInsect,
    default_config,
)


# ============================================================================ #
#                                CLI arguments                                 #
# ============================================================================ #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory produced by undirected_locomotion_ppo_jax.py.",
    )
    parser.add_argument(
        "--grid", type=int, default=25, help="Grid resolution per PCA axis."
    )
    parser.add_argument(
        "--range",
        type=float,
        default=2.0,
        dest="pc_range",
        help="How many standard deviations along each PC to sweep.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=500,
        help="Rollout length (control steps) for each grid point.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--center",
        choices=["final", "best"],
        default="final",
        help="Which snapshot to center the PCA plane on: the final params, or "
        "the snapshot with the best eval reward.",
    )
    parser.add_argument(
        "--basis",
        choices=["pca", "random", "both"],
        default="pca",
        help="Direction basis for the landscape slice. `pca` uses the top two "
        "trajectory PCs; `random` uses two filter-normalized Gaussian directions "
        "(Li et al. 2018 style); `both` produces plots for each.",
    )
    parser.add_argument(
        "--filter-normalize",
        dest="filter_normalize",
        action="store_true",
        help="Scale each kernel column of each direction to match the "
        "corresponding column norm in the center weights; zero biases. "
        "Enabled by default; this flag is a no-op (kept for clarity).",
    )
    parser.add_argument(
        "--no-filter-normalize",
        dest="filter_normalize",
        action="store_false",
        help="Disable filter-wise normalization. Reproduces the pre-paper "
        "behavior where PC directions are used as raw unit vectors.",
    )
    parser.set_defaults(filter_normalize=True)
    parser.add_argument(
        "--random-scale",
        type=float,
        default=0.3,
        help="For --basis random: half-width of the sweep as a fraction of "
        "||θ_c|| (after filter normalization). Default 0.3 ≈ ±0.3 weight-norm.",
    )
    parser.add_argument(
        "--seeds-per-cell",
        type=int,
        default=1,
        help="Number of reset seeds to average per grid cell. Yields a noise "
        "band (std heatmap) when >1.",
    )
    parser.add_argument(
        "--reward",
        action="store_true",
        help="Also plot the mean per-step reward surface (what PPO actually "
        "optimises), alongside displacement.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Produce a GIF showing the snapshot trajectory across the landscape.",
    )
    parser.add_argument(
        "--fps", type=int, default=4, help="Frames per second for the animation."
    )
    return parser.parse_args()


# ============================================================================ #
#                               Env + networks                                 #
# ============================================================================ #
def _build_eval_env(
    num_envs: int, morphology: str = "insect", n_pairs: int = 2,
) -> UndirectedLocomotionInsect:
    """Env sized for ``num_envs`` concurrent rollouts (used with jax.vmap).

    ``morphology`` and ``n_pairs`` must match what the policy was trained on,
    otherwise the saved observation normalizer's mean/std will have the wrong
    shape and ``preprocess_observations_fn`` will fail with a broadcast error.
    """
    cfg = default_config()
    cfg.morphology = morphology
    cfg.n_pairs = n_pairs
    cfg.naconmax = num_envs * NACONMAX_PER_ENV
    cfg.naccdmax = num_envs * NACCDMAX_PER_ENV
    return UndirectedLocomotionInsect(cfg)


def _build_inference_factory(env: UndirectedLocomotionInsect, ppo_cfg: dict) -> Any:
    network_factory_kwargs = ppo_cfg["network_factory"]
    normalize_observations = ppo_cfg.get("normalize_observations", False)
    normalize = (
        running_statistics.normalize
        if normalize_observations
        else (lambda x, y: x)
    )
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
        **network_factory_kwargs,
    )
    return ppo_networks.make_inference_fn(ppo_network)


# ============================================================================ #
#                              Landscape eval                                  #
# ============================================================================ #
def _flatten_policy(policy_params: Any) -> np.ndarray:
    flat, _ = ravel_pytree(policy_params)
    return np.asarray(flat)


# ---------------------------------------------------------------------------- #
# Filter-wise normalization (Li, Xu, Taylor, Studer, Goldstein, 2018).
#
# For each weight matrix (`kernel`) in the policy MLP, column j is the
# fan-in vector feeding output neuron j — the direct analogue of a conv
# filter. The paper rescales each filter of a direction d_{i,j} to match
# the corresponding filter norm in the center weights W_{i,j}:
#
#     d_{i,j}  ←  (d_{i,j} / ||d_{i,j}||) · ||W_{i,j}||
#
# Biases are zeroed because they lack a scale-invariance to normalise
# against. The result: one unit of α along the normalized direction
# perturbs every layer by a consistent *fraction of that layer's own
# weight norm*, so sharpness comparisons across checkpoints/runs become
# meaningful.
# ---------------------------------------------------------------------------- #
def _filter_normalize_pytree(direction_tree: Any, center_tree: Any) -> Any:
    """Return a pytree with the same structure as ``direction_tree`` whose
    kernel columns match the column norms of ``center_tree`` and whose
    bias/scalar leaves are zero.
    """

    def _norm(d_leaf: jax.Array, c_leaf: jax.Array) -> jax.Array:
        if d_leaf.ndim == 2:
            d_norms = jp.linalg.norm(d_leaf, axis=0, keepdims=True) + 1e-10
            c_norms = jp.linalg.norm(c_leaf, axis=0, keepdims=True)
            return d_leaf / d_norms * c_norms
        # Biases and any other 1D/0D leaves: no direction (frozen).
        return jp.zeros_like(d_leaf)

    return jax.tree_util.tree_map(_norm, direction_tree, center_tree)


def _sample_random_direction_tree(rng: jax.Array, template_tree: Any) -> Any:
    """Draw a Gaussian direction whose leaves match ``template_tree`` shapes."""
    leaves, treedef = jax.tree_util.tree_flatten(template_tree)
    keys = jax.random.split(rng, len(leaves))
    sampled = [
        jax.random.normal(k, shape=l.shape, dtype=l.dtype)
        for k, l in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, sampled)


def _trajectory_scale_in_direction(
    direction_flat: np.ndarray,
    all_weights: np.ndarray,
    center_flat: np.ndarray,
) -> float:
    """std of the training trajectory projected onto ``direction_flat``."""
    proj = (all_weights - center_flat[None, :]) @ direction_flat
    return float(np.std(proj))


def _build_basis(
    *,
    basis_name: str,
    center_policy_tree: Any,
    center_policy_flat: np.ndarray,
    all_weights: np.ndarray,
    pca: PCA | None,
    unravel_fn: Any,
    filter_normalize: bool,
    random_scale: float,
    pc_range: float,
    rng_seed: int,
) -> dict[str, Any]:
    """Produce two basis directions + sweep extents + metadata.

    Returns a dict with keys:
      d1_flat, d2_flat  — (d,) float32 directions.
      s1, s2            — half-width of α/β sweep (meters along direction).
      label1, label2    — axis labels for plots.
      kind              — "pca" or "random", echoed back for titles/filenames.
      meta              — free-form info dict for logging (e.g. filter_normalize).
    """
    if basis_name == "pca":
        assert pca is not None, "PCA must be fit for basis='pca'"
        d1 = pca.components_[0].astype(np.float32)
        d2 = pca.components_[1].astype(np.float32)
    elif basis_name == "random":
        rng = jax.random.PRNGKey(rng_seed)
        r1, r2 = jax.random.split(rng, 2)
        d1_tree = _sample_random_direction_tree(r1, center_policy_tree)
        d2_tree = _sample_random_direction_tree(r2, center_policy_tree)
        d1 = np.asarray(ravel_pytree(d1_tree)[0], dtype=np.float32)
        d2 = np.asarray(ravel_pytree(d2_tree)[0], dtype=np.float32)
    else:
        raise ValueError(f"unknown basis_name {basis_name!r}")

    if filter_normalize:
        d1_tree = unravel_fn(jp.asarray(d1))
        d2_tree = unravel_fn(jp.asarray(d2))
        d1 = np.asarray(
            ravel_pytree(_filter_normalize_pytree(d1_tree, center_policy_tree))[0],
            dtype=np.float32,
        )
        d2 = np.asarray(
            ravel_pytree(_filter_normalize_pytree(d2_tree, center_policy_tree))[0],
            dtype=np.float32,
        )

    # Standardize both directions to unit ℓ² length. Filter normalization
    # imposes a *per-layer ratio* (each layer's share of the perturbation
    # proportional to that layer's own weight norm), which is preserved by
    # uniform rescaling. Without this, the filter-normalized direction has
    # ‖d‖ ≈ ‖θ_c‖, which — combined with the trajectory-std scaling below —
    # inflates the sweep by ‖θ_c‖² and the whole grid collapses onto a dot
    # at the center with everything else being far-field garbage.
    d1 = d1 / (float(np.linalg.norm(d1)) + 1e-12)
    d2 = d2 / (float(np.linalg.norm(d2)) + 1e-12)

    center_norm = float(np.linalg.norm(center_policy_flat))

    if basis_name == "pca":
        # Sweep is ±pc_range × std of trajectory projection onto the unit
        # direction. With unit ‖d‖, this is a weight-space ℓ² distance;
        # pc_range=2 covers ~95% of the trajectory projection along d.
        s1 = pc_range * _trajectory_scale_in_direction(d1, all_weights, center_policy_flat)
        s2 = pc_range * _trajectory_scale_in_direction(d2, all_weights, center_policy_flat)
        label1 = "α (PC1 traj std units)"
        label2 = "β (PC2 traj std units)"
    else:
        # Random directions have no trajectory to measure. Follow Li et al.
        # 2018 and sweep ±random_scale × ‖θ_c‖ along each unit direction.
        s1 = random_scale * center_norm
        s2 = random_scale * center_norm
        label1 = f"α (×‖θ_c‖ along rand-d1)"
        label2 = f"β (×‖θ_c‖ along rand-d2)"

    # Orthogonality / scale diagnostics.
    cos12 = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))
    meta = {
        "filter_normalize": filter_normalize,
        "cos(d1,d2)": cos12,
        "||d1||": float(np.linalg.norm(d1)),
        "||d2||": float(np.linalg.norm(d2)),
        "||θ_c||": center_norm,
    }

    return {
        "d1_flat": d1, "d2_flat": d2,
        "s1": float(s1), "s2": float(s2),
        "label1": label1, "label2": label2,
        "kind": basis_name,
        "meta": meta,
    }


# ============================================================================ #
#                            Joint grouping + wave                             #
# ============================================================================ #
def _group_joints(mj_model: Any, morphology: str) -> dict[str, list[int]]:
    """Group actuated joints into spine / right-legs / left-legs, ordered
    front→back.

    Values are indices into ``qpos[7:]`` (equivalently, into the action
    vector), not joint ids in the full ``mj_model``.
    """
    nu = int(mj_model.nu)
    names: list[str] = []
    for i in range(nu):
        jid = int(mj_model.actuator(i).trnid[0])
        names.append(mj_model.joint(jid).name)

    buckets: dict[str, list[tuple[int, int]]] = {"spine": [], "right": [], "left": []}

    if morphology == "insect":
        # insect_small: `hinge_{0..7}servo`.
        #   0, 1 → spine (0 back near core, 1 front near brick_1)
        #   2, 4, 6 → right legs (front, middle, rear)
        #   3, 5, 7 → left legs  (front, middle, rear)
        for i, name in enumerate(names):
            m = re.search(r"hinge_(\d+)", name)
            if m is None:
                continue
            n = int(m.group(1))
            if n == 1:
                buckets["spine"].append((0, i))
            elif n == 0:
                buckets["spine"].append((1, i))
            elif n in (2, 4, 6):
                buckets["right"].append(((n - 2) // 2, i))
            elif n in (3, 5, 7):
                buckets["left"].append(((n - 3) // 2, i))
    else:
        # Hierarchical names from `_attach(..., "RH"|"LH"|"H{i}")`, e.g.
        # `robot1/C-H0-B0-H1-B1-LH-servo`. Segment index = number of H{i}
        # tokens on the chain before the terminal module.
        for i, name in enumerate(names):
            tokens = [t for t in re.split(r"[-/]", name) if t]
            h_count = sum(1 for t in tokens if re.fullmatch(r"H\d+", t))
            suffix = None
            for t in reversed(tokens):
                if re.fullmatch(r"RH|LH|H\d+", t):
                    suffix = t
                    break
            if suffix is None:
                continue
            if suffix == "RH":
                buckets["right"].append((h_count, i))
            elif suffix == "LH":
                buckets["left"].append((h_count, i))
            else:
                buckets["spine"].append((int(suffix[1:]), i))

    # Front→back sort. For insect_small the key already encodes that;
    # for hierarchical names higher H-count = closer to tip = front.
    reverse = morphology != "insect"
    out: dict[str, list[int]] = {}
    for key, items in buckets.items():
        items.sort(key=lambda x: x[0], reverse=reverse)
        out[key] = [i for _, i in items]
    return out


def _side_wave_score(signals: np.ndarray, dt: float, min_freq_hz: float = 0.5) -> tuple[float, float, float]:
    """FFT-based metachronal wave score for one side of the body.

    Parameters
    ----------
    signals : (T, J) joint angles, columns ordered front→back.
    dt      : seconds between samples (ctrl_dt).

    Returns
    -------
    (r2, prominence, dom_freq_hz)
      r2          — coefficient of determination of a linear phase-vs-segment fit.
      prominence  — fraction of above-``min_freq_hz`` energy concentrated at
                    the side-wide dominant frequency bin.
      dom_freq_hz — the dominant frequency itself (NaN if the signal is dead).
    """
    T, J = signals.shape
    if J < 2 or T < 8 or not np.isfinite(signals).all():
        return 0.0, 0.0, float("nan")

    x = signals - signals.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(T, d=dt)
    power = np.sum(np.abs(spec) ** 2, axis=1)

    mask = freqs >= min_freq_hz
    if not mask.any() or power[mask].sum() == 0.0:
        return 0.0, 0.0, float("nan")

    k = int(np.argmax(np.where(mask, power, -1.0)))
    prominence = float(power[k] / (power[mask].sum() + 1e-12))

    phases = np.unwrap(np.angle(spec[k]))
    xs = np.arange(J, dtype=np.float64)
    slope, intercept = np.polyfit(xs, phases, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((phases - yhat) ** 2))
    ss_tot = float(np.sum((phases - phases.mean()) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    return r2, prominence, float(freqs[k])


def _wave_metric_landscape(
    qpos_series: np.ndarray,
    groups: dict[str, list[int]],
    dt: float,
) -> np.ndarray:
    """Per-grid-point wave metric = mean over (R, L) sides of R² × prominence.

    ``qpos_series`` has shape (N, T, n_joints). Returns an (N,) float array;
    NaN for trajectories where neither side yielded a usable score.
    """
    N = qpos_series.shape[0]
    scores = np.full(N, np.nan, dtype=np.float64)
    r_idx = groups.get("right", [])
    l_idx = groups.get("left", [])
    for n in range(N):
        side_scores: list[float] = []
        for idx in (r_idx, l_idx):
            if len(idx) >= 2:
                r2, prom, _ = _side_wave_score(qpos_series[n][:, idx], dt)
                side_scores.append(r2 * prom)
        if side_scores:
            scores[n] = float(np.mean(side_scores))
    return scores


def build_landscape(
    env: UndirectedLocomotionInsect,
    inference_factory: Any,
    final_params: tuple,
    center_policy_flat: np.ndarray,
    basis: dict[str, Any],
    grid_n: int,
    episode_length: int,
    seed: int,
    seeds_per_cell: int,
) -> dict[str, np.ndarray]:
    """Evaluate a grid of perturbed policies around ``center_policy_flat``.

    Keeps normalizer + value params fixed at the final-training values and
    only perturbs policy_params along ``basis['d1_flat'], basis['d2_flat']``,
    with half-widths ``basis['s1'], basis['s2']``. Returns a dict with:

      A, B                — (grid_n, grid_n) grid coords.
      alphas, betas       — the 1D axis arrays.
      Z_disp_mean/_std    — displacement across reset seeds.
      Z_reward_mean/_std  — mean per-step reward across reset seeds.
      qpos_series         — (grid_n*grid_n, T, nu) joint angles from seed 0.
      action_series       — (grid_n*grid_n, T, nu) policy commands from seed 0.
    """
    final_normalizer, final_policy, final_value = final_params
    _, unravel_fn = ravel_pytree(final_policy)

    d1 = basis["d1_flat"]
    d2 = basis["d2_flat"]
    s1 = basis["s1"]
    s2 = basis["s2"]

    alphas = np.linspace(-s1, s1, grid_n)
    betas = np.linspace(-s2, s2, grid_n)

    grid_flat = np.stack(
        [center_policy_flat + a * d1 + b * d2 for a in alphas for b in betas]
    ).astype(np.float32)

    reset_rngs = jax.random.split(
        jax.random.PRNGKey(seed + 100), seeds_per_cell
    )
    scan_rng = jax.random.PRNGKey(seed + 101)

    def _single_rollout(policy_flat: jax.Array, reset_rng: jax.Array):
        policy_params = unravel_fn(policy_flat)
        params = (final_normalizer, policy_params, final_value)
        policy = inference_factory(params, deterministic=True)
        state = env.reset(reset_rng)

        def body(carry, _):
            state, rng = carry
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
            new_state = env.step(state, action)
            joint_angles = new_state.data.qpos[7:]
            # new_state.reward is per-step scalar; sum→episode_reward,
            # mean→mean per-step reward (the quantity PPO's advantage targets).
            return (new_state, rng), (joint_angles, action, new_state.reward)

        (final_state, _), (joints_t, actions_t, rewards_t) = jax.lax.scan(
            body, (state, scan_rng), None, length=episode_length
        )
        disp = jp.linalg.norm(final_state.data.qpos[:2])
        disp = jp.where(jp.isnan(disp), 0.0, disp)
        mean_reward = jp.where(
            jp.isnan(rewards_t).any(), 0.0, jp.mean(rewards_t)
        )
        return disp, mean_reward, joints_t, actions_t

    # Outer vmap: grid points; inner vmap: reset seeds.
    def _multi_seed(policy_flat: jax.Array):
        return jax.vmap(lambda rk: _single_rollout(policy_flat, rk))(reset_rngs)

    batched = jax.jit(jax.vmap(_multi_seed))

    print(
        f"Evaluating {grid_n * grid_n} grid points × {seeds_per_cell} seeds on "
        f"{jax.default_backend()}... (first call compiles the batched rollout)"
    )
    t0 = time.monotonic()
    disp_arr, reward_arr, joints_arr, actions_arr = batched(jp.asarray(grid_flat))
    # Shapes: (grid², seeds), (grid², seeds), (grid², seeds, T, nu), …
    disp_np = np.asarray(disp_arr)         # (N, S)
    reward_np = np.asarray(reward_arr)     # (N, S)
    # Keep joint/action series only for seed 0 — they're used by the wave
    # analysis which needs the gait trajectory, not an average of seeds.
    joints_np = np.asarray(joints_arr[:, 0])    # (N, T, nu)
    actions_np = np.asarray(actions_arr[:, 0])  # (N, T, nu)
    print(f"  landscape eval done in {time.monotonic() - t0:.1f} s")

    A, B = np.meshgrid(alphas, betas, indexing="ij")

    def _reshape_stat(arr_ns: np.ndarray, agg: str) -> np.ndarray:
        if agg == "mean":
            vals = arr_ns.mean(axis=1)
        else:
            vals = arr_ns.std(axis=1)
        return vals.reshape(grid_n, grid_n)

    return {
        "A": A, "B": B,
        "alphas": alphas, "betas": betas,
        "Z_disp_mean": _reshape_stat(disp_np, "mean"),
        "Z_disp_std": _reshape_stat(disp_np, "std"),
        "Z_reward_mean": _reshape_stat(reward_np, "mean"),
        "Z_reward_std": _reshape_stat(reward_np, "std"),
        "qpos_series": joints_np,
        "action_series": actions_np,
    }


# ============================================================================ #
#                           Hessian / sharpness                                #
# ============================================================================ #
def hessian_at_center(
    Z: np.ndarray, dalpha: float, dbeta: float,
) -> dict[str, float]:
    """2×2 finite-difference Hessian of the landscape at (0, 0).

    The paper (Li et al. 2018, §6) proposes a "convexity index"
    c = |λ_min_neg / λ_max_pos| ∈ [-1, 0], defined only when the Hessian
    has both a positive and a negative eigenvalue (saddle). For pure
    basins (both λ > 0) or pure peaks (both λ < 0) that index is
    undefined; we return ``shape`` to say which case we're in and leave
    ``convexity`` NaN there, so the title never carries a nonsensical
    number.

    ``shape`` values:
      "basin"  — both λ > 0  (local min of Z)
      "peak"   — both λ < 0  (local max of Z)
      "saddle" — mixed signs
      "flat"   — both ≈ 0
    """
    gn = Z.shape[0]
    c = gn // 2
    if gn < 3 or c == 0 or c == gn - 1:
        return {
            "lam_max": float("nan"), "lam_min": float("nan"),
            "convexity": float("nan"), "shape": "n/a",
            "f_center": float("nan"),
        }

    f_cc = float(Z[c, c])
    f_pc = float(Z[c + 1, c]); f_mc = float(Z[c - 1, c])
    f_cp = float(Z[c, c + 1]); f_cm = float(Z[c, c - 1])
    f_pp = float(Z[c + 1, c + 1]); f_mm = float(Z[c - 1, c - 1])
    f_pm = float(Z[c + 1, c - 1]); f_mp = float(Z[c - 1, c + 1])

    f_aa = (f_pc - 2 * f_cc + f_mc) / (dalpha ** 2 + 1e-18)
    f_bb = (f_cp - 2 * f_cc + f_cm) / (dbeta ** 2 + 1e-18)
    f_ab = (f_pp - f_pm - f_mp + f_mm) / (4 * dalpha * dbeta + 1e-18)
    H = np.array([[f_aa, f_ab], [f_ab, f_bb]])
    eigs = np.linalg.eigvalsh(H)
    lam_min, lam_max = float(eigs[0]), float(eigs[-1])

    scale = max(abs(lam_max), abs(lam_min), 1e-12)
    rel_tol = 1e-6
    pos_max = lam_max > rel_tol * scale
    neg_min = lam_min < -rel_tol * scale
    if pos_max and neg_min:
        shape = "saddle"
        # Signed Li et al. index: how much negative curvature vs positive.
        convexity = -abs(lam_min) / abs(lam_max)
    elif pos_max and not neg_min:
        shape = "basin"
        convexity = float("nan")  # convexity undefined without a negative λ
    elif neg_min and not pos_max:
        shape = "peak"
        convexity = float("nan")  # peak = concave; "convexity" is misleading
    else:
        shape = "flat"
        convexity = float("nan")
    return {
        "lam_max": lam_max, "lam_min": lam_min,
        "convexity": convexity, "shape": shape,
        "f_center": f_cc,
    }


# ============================================================================ #
#                                   Plots                                      #
# ============================================================================ #
def plot_scree(evr: np.ndarray, output_path: Path) -> None:
    cumvar = np.cumsum(evr)
    fig, (ax_ind, ax_cum) = plt.subplots(1, 2, figsize=(12, 4))

    n_show = min(30, len(evr))
    ax_ind.bar(range(1, n_show + 1), evr[:n_show] * 100, color="steelblue")
    ax_ind.axvline(2.5, color="red", linestyle="--", label="2-PC cutoff")
    ax_ind.set_xlabel("Principal component")
    ax_ind.set_ylabel("Explained variance (%)")
    ax_ind.set_title(f"Individual explained variance (top {n_show} PCs)")
    ax_ind.legend()

    ax_cum.plot(range(1, len(evr) + 1), cumvar * 100, color="steelblue")
    ax_cum.axhline(95, color="orange", linestyle="--", label="95%")
    ax_cum.axhline(99, color="red", linestyle="--", label="99%")
    ax_cum.axvline(2, color="grey", linestyle=":", label="2 PCs shown in landscape")
    ax_cum.set_xlabel("Number of principal components")
    ax_cum.set_ylabel("Cumulative explained variance (%)")
    ax_cum.set_title("Cumulative explained variance")
    ax_cum.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Scree plot saved to {output_path}")


def plot_reward(history: dict, output_path: Path) -> None:
    steps = np.array(history["steps"])
    reward = np.array(history["reward"])
    reward_std = np.array(history["reward_std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, reward, color="darkorange", linewidth=1.5)
    ax.fill_between(
        steps, reward - reward_std, reward + reward_std, color="darkorange", alpha=0.25
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval episode reward")
    ax.set_title("PPO eval reward over training (one snapshot per point)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Reward curve saved to {output_path}")


def plot_landscape(
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    output_path: Path,
    label: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    subtitle: str = "",
) -> None:
    """3D surface + 2D contour of a scalar landscape over the (α, β) grid.

    ``subtitle`` is appended on a second line and is a natural slot for
    diagnostics (Hessian spectrum, PC variance, cos alignment).
    """
    center_x, center_y = 0.0, 0.0

    fig = plt.figure(figsize=(14, 6))
    header = title if not subtitle else f"{title}\n{subtitle}"

    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(A, B, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, label=zlabel)
    ax3d.scatter(
        center_x, center_y, np.nanmax(Z),
        color="red", s=60, zorder=5, label=label,
    )
    ax3d.set_xlabel(xlabel)
    ax3d.set_ylabel(ylabel)
    ax3d.set_zlabel(zlabel)
    ax3d.set_title(header)
    ax3d.legend()

    ax2d = fig.add_subplot(122)
    hm = ax2d.contourf(A, B, Z, levels=30, cmap=cm.viridis)
    fig.colorbar(hm, ax=ax2d, label=zlabel)
    ax2d.scatter(
        center_x, center_y,
        color="red", s=60, zorder=5, label=label,
    )
    ax2d.set_xlabel(xlabel)
    ax2d.set_ylabel(ylabel)
    ax2d.set_title("top view")
    ax2d.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Landscape saved to {output_path}")


def plot_1d_slices(
    alphas: np.ndarray,
    betas: np.ndarray,
    Z_mean: np.ndarray,
    Z_std: np.ndarray | None,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    title: str,
    subtitle: str = "",
) -> None:
    """Cross-section f(α, 0) and f(0, β) on shared y-axis.

    Where sharpness-vs-flatness between runs becomes directly comparable,
    following Li et al. 2018 §5. ``Z_std`` (if provided from multi-seed eval)
    is drawn as a ±1σ band around the mean curve.
    """
    gn = Z_mean.shape[0]
    c = gn // 2
    f_alpha = Z_mean[:, c]
    f_beta = Z_mean[c, :]
    s_alpha = Z_std[:, c] if Z_std is not None else None
    s_beta = Z_std[c, :] if Z_std is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    header = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(header)

    axes[0].plot(alphas, f_alpha, color="tab:blue", linewidth=1.6)
    if s_alpha is not None:
        axes[0].fill_between(
            alphas, f_alpha - s_alpha, f_alpha + s_alpha,
            color="tab:blue", alpha=0.2, label="±1σ (reset seeds)",
        )
    axes[0].axvline(0.0, color="red", linestyle="--", alpha=0.4, label="center")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("f(α, 0)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(betas, f_beta, color="tab:orange", linewidth=1.6)
    if s_beta is not None:
        axes[1].fill_between(
            betas, f_beta - s_beta, f_beta + s_beta,
            color="tab:orange", alpha=0.2,
        )
    axes[1].axvline(0.0, color="red", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("β")
    axes[1].set_title("f(0, β)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"1D cross-sections saved to {output_path}")


def plot_std_heatmap(
    A: np.ndarray,
    B: np.ndarray,
    Z_std: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    hm = ax.contourf(A, B, Z_std, levels=30, cmap=cm.magma)
    fig.colorbar(hm, ax=ax, label="std across reset seeds (m)")
    ax.scatter(0.0, 0.0, color="white", edgecolors="black", s=70, zorder=5, label="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Std heatmap saved to {output_path}")


def plot_wave_landscape(
    A: np.ndarray,
    B: np.ndarray,
    Z_wave: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(Z_wave)
    hm = ax.contourf(A, B, masked, levels=30, cmap=cm.plasma, vmin=0.0, vmax=1.0)
    fig.colorbar(hm, ax=ax, label="Wave coherence  (R² × spectral prominence)")
    ax.scatter(0.0, 0.0, color="white", edgecolors="black", s=70, zorder=5, label="Center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Metachronal wave coherence (qpos)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wave landscape saved to {output_path}")


def _select_picks(
    Z_disp: np.ndarray, Z_wave: np.ndarray, grid_n: int,
) -> list[tuple[str, int, int]]:
    """Return (label, i, j) grid cells to show detail plots for."""
    picks: list[tuple[str, int, int]] = []
    # Center cell.
    c = grid_n // 2
    picks.append(("center", c, c))
    # Max-displacement cell.
    max_flat = int(np.nanargmax(Z_disp))
    picks.append(("max_disp", max_flat // grid_n, max_flat % grid_n))
    # Min-displacement cell (lowest finite value).
    disp_for_min = np.where(np.isfinite(Z_disp), Z_disp, np.inf)
    min_flat = int(np.argmin(disp_for_min))
    picks.append(("min_disp", min_flat // grid_n, min_flat % grid_n))
    # Max-wave-coherence cell (if any valid).
    if np.isfinite(Z_wave).any():
        wave_flat = int(np.nanargmax(Z_wave))
        picks.append(("max_wave", wave_flat // grid_n, wave_flat % grid_n))
    else:
        picks.append(("corner", 0, 0))
    # Deduplicate (center can collide with max/min).
    seen: set[tuple[int, int]] = set()
    uniq: list[tuple[str, int, int]] = []
    for label, i, j in picks:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        uniq.append((label, i, j))
    return uniq


def plot_wave_picks(
    picks: list[tuple[str, int, int]],
    qpos_series: np.ndarray,     # (grid, grid, T, nu)
    action_series: np.ndarray,   # (grid, grid, T, nu)
    groups: dict[str, list[int]],
    ctrl_dt: float,
    A: np.ndarray,
    B: np.ndarray,
    Z_disp: np.ndarray,
    Z_wave: np.ndarray,
    output_path: Path,
    action_scale: float,
) -> None:
    """Spatiotemporal heatmaps + per-side wave numbers for a handful of
    marked grid points.

    Rows: qpos (actual joint angle, rad) and action × action_scale (commanded
    target, rad). Columns: spine, right legs, left legs. Columns with <1 joint
    are hidden.
    """
    group_order = [k for k in ("spine", "right", "left") if len(groups[k]) >= 1]
    if not group_order:
        print("No joints identified for any body region — skipping pick plots.")
        return

    T = qpos_series.shape[2]
    t = np.arange(T) * ctrl_dt

    n_picks = len(picks)
    n_cols = len(group_order)
    fig = plt.figure(figsize=(3.6 * n_cols, 3.2 * n_picks * 2))
    gs = GridSpec(
        n_picks * 2, n_cols + 1,
        width_ratios=[*([1.0] * n_cols), 0.05],
        hspace=0.55, wspace=0.25,
    )

    # Shared colour scale for qpos vs. action (both ~rad after action_scale).
    all_qpos = np.concatenate([qpos_series[i, j].ravel() for _, i, j in picks])
    all_act = np.concatenate(
        [(action_series[i, j] * action_scale).ravel() for _, i, j in picks]
    )
    vmax = float(np.nanpercentile(np.abs(np.concatenate([all_qpos, all_act])), 99.0))
    if not np.isfinite(vmax) or vmax < 1e-6:
        vmax = float(action_scale)

    for row, (label, i, j) in enumerate(picks):
        qpos_t = qpos_series[i, j]              # (T, nu)
        act_t = action_series[i, j] * action_scale  # commanded targets, rad

        # Per-side wave scores for the pick's title.
        side_scores = []
        for side in ("right", "left"):
            if len(groups[side]) >= 2:
                r2, prom, freq = _side_wave_score(qpos_t[:, groups[side]], ctrl_dt)
                side_scores.append(f"{side[0].upper()}: R²={r2:.2f} prom={prom:.2f} f={freq:.2f}Hz")
            else:
                side_scores.append(f"{side[0].upper()}: n/a")
        wave_val = Z_wave[i, j]
        wave_str = f"{wave_val:.3f}" if np.isfinite(wave_val) else "n/a"
        header = (
            f"[{label}]  (α={A[i, j]:+.3f}, β={B[i, j]:+.3f})   "
            f"disp={Z_disp[i, j]:.2f} m   wave={wave_str}\n"
            + "   |   ".join(side_scores)
        )

        for col, side in enumerate(group_order):
            idx = groups[side]
            if not idx:
                continue
            # qpos row
            ax_q = fig.add_subplot(gs[row * 2, col])
            # action row
            ax_a = fig.add_subplot(gs[row * 2 + 1, col])

            def _render(ax, data_t, title):
                if len(idx) == 1:
                    ax.plot(t, data_t[:, idx[0]], color="steelblue")
                    ax.set_xlabel("time (s)")
                    ax.set_ylabel("angle (rad)")
                    ax.set_title(title)
                    return None
                im = ax.imshow(
                    data_t[:, idx].T,
                    aspect="auto", origin="lower",
                    extent=(0.0, t[-1], -0.5, len(idx) - 0.5),
                    cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax,
                )
                ax.set_yticks(np.arange(len(idx)))
                ax.set_yticklabels([f"{side[0].upper()}{k}" for k in range(len(idx))])
                ax.set_xlabel("time (s)")
                ax.set_title(title)
                return im

            _render(ax_q, qpos_t, f"{side} qpos  ({label})" if col == 0 else f"{side} qpos")
            _render(ax_a, act_t, f"{side} action·π/2" if col == 0 else f"{side} action·π/2")

            if col == 0:
                ax_q.text(
                    -0.35, 1.08, header, transform=ax_q.transAxes,
                    fontsize=9, ha="left", va="bottom",
                )

        # Shared colour-bar at far right of each pick's two rows.
        cbar_ax = fig.add_subplot(gs[row * 2 : row * 2 + 2, -1])
        sm = cm.ScalarMappable(cmap=cm.RdBu_r)
        sm.set_array(np.array([-vmax, vmax]))
        sm.set_clim(-vmax, vmax)
        fig.colorbar(sm, cax=cbar_ax, label="rad")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wave picks saved to {output_path}")


def animate_migration(
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    xlabel: str,
    ylabel: str,
    all_weights: np.ndarray,
    rewards_per_snapshot: np.ndarray,
    d1_flat: np.ndarray,
    d2_flat: np.ndarray,
    center_policy_flat: np.ndarray,
    output_path: Path,
    fps: int,
) -> None:
    """Snapshot-by-snapshot animation of the policy drifting across the landscape.

    Projects each snapshot θ_k onto the (possibly non-orthogonal, possibly
    filter-normalized) basis directions:

        xs[k] = ⟨d1, θ_k − θ_c⟩ / ‖d1‖²
        ys[k] = ⟨d2, θ_k − θ_c⟩ / ‖d2‖²

    which, for a non-orthonormal basis, gives the oblique-projection
    coefficient that satisfies α·d1 + β·d2 ≈ projection into span(d1, d2).
    """
    delta = all_weights - center_policy_flat[None, :]
    xs = delta @ d1_flat / (float(np.dot(d1_flat, d1_flat)) + 1e-12)
    ys = delta @ d2_flat / (float(np.dot(d2_flat, d2_flat)) + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(A, B, Z, levels=30, cmap=cm.viridis)
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(Z)
    fig.colorbar(mappable, ax=ax, label="Final displacement (m)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    trail, = ax.plot([], [], color="white", linewidth=1.2, alpha=0.7)
    dot = ax.scatter([], [], color="red", s=80, zorder=6)
    title = ax.set_title("")

    def init():
        trail.set_data([], [])
        dot.set_offsets(np.empty((0, 2)))
        return trail, dot

    def update(frame: int):
        trail.set_data(xs[: frame + 1], ys[: frame + 1])
        dot.set_offsets([[xs[frame], ys[frame]]])
        reward = rewards_per_snapshot[frame]
        reward_str = f"{reward:+.2f}" if np.isfinite(reward) else "n/a"
        title.set_text(
            f"Snapshot {frame + 1}/{len(xs)}  |  eval reward: {reward_str}"
        )
        return trail, dot, title

    anim = FuncAnimation(
        fig, update, frames=len(xs), init_func=init, blit=True, interval=1000 // fps
    )
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Migration animation saved to {output_path}")


# ============================================================================ #
#                                    Main                                      #
# ============================================================================ #
def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    # ---- Load artefacts --------------------------------------------------- #
    ppo_cfg = json.loads((run_dir / "ppo_config.json").read_text())
    env_cfg_path = run_dir / "env_config.json"
    env_cfg = json.loads(env_cfg_path.read_text()) if env_cfg_path.exists() else {}

    # Auto-detect CPG runs (their env_config.json contains a `cpg` sub-config)
    # and swap in the CPG env class + default_config at module scope. Without
    # this, _build_eval_env would construct the baseline env whose obs/action
    # dims don't match the saved normalizer and policy, producing a broadcast
    # error on the first policy() call.
    if "cpg" in env_cfg:
        global UndirectedLocomotionInsect, default_config  # noqa: PLW0603
        from undirected_locomotion_ppo_jax_cpg import (  # noqa: E402
            CPGUndirectedLocomotionInsect as UndirectedLocomotionInsect,
            default_config,
        )
        print("Detected CPG env config — using CPG env class for eval")

    morphology = env_cfg.get("morphology") or "insect"
    n_pairs = env_cfg.get("n_pairs") or 2
    print(f"env morphology={morphology!r}  n_pairs={n_pairs}")
    final_params = model.load_params(str(run_dir / "params.pkl"))
    snapshots_path = run_dir / "params_snapshots.pkl"
    if not snapshots_path.exists():
        raise FileNotFoundError(
            f"No params_snapshots.pkl in {run_dir}. Re-train with the current "
            "undirected_locomotion_ppo_jax.py (it saves snapshots per eval)."
        )
    snapshots = model.load_params(str(snapshots_path))
    history_path = run_dir / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None

    print(f"Loaded {len(snapshots)} snapshots from {run_dir}")

    # ---- Flatten the policy_params trajectory ----------------------------- #
    all_weights = np.stack([_flatten_policy(sn[1][1]) for sn in snapshots])
    final_policy_flat = _flatten_policy(final_params[1])
    print(
        f"Policy params dim: {all_weights.shape[1]}  |  "
        f"snapshots: {all_weights.shape[0]}"
    )

    # ---- Pick landscape center ------------------------------------------- #
    center_label = "Final policy"
    center_policy_flat = final_policy_flat
    if args.center == "best" and history is not None and history["reward"]:
        rewards = np.array(history["reward"])
        best_i = int(np.nanargmax(rewards))
        if best_i < len(snapshots):
            center_policy_flat = all_weights[best_i]
            center_label = f"Best snapshot (step {snapshots[best_i][0]})"
            print(f"Centering on {center_label}, reward={rewards[best_i]:+.3f}")

    # ---- PCA (full for scree, 2 PCs for the plane) ----------------------- #
    pca_full = PCA().fit(all_weights)
    evr_full = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr_full)
    print(
        f"PC1: {evr_full[0] * 100:.1f}%  PC2: {evr_full[1] * 100:.1f}%  "
        f"(cumvar at 2 PCs: {cumvar[1] * 100:.1f}%)"
    )
    plot_scree(evr_full, run_dir / "landscape_ppo_scree.png")
    if history is not None:
        plot_reward(history, run_dir / "landscape_ppo_reward_curve.png")

    pca = PCA(n_components=2).fit(all_weights)

    # ---- Trajectory alignment (final descent step vs. the 2 PCs) --------- #
    # If the plane captures the bulk of the trajectory but misses the final
    # polish direction, the surface near the center won't reflect where PPO
    # was pointing on the last update. Report cos(θ_K − θ_{K-1}, v_i).
    if all_weights.shape[0] >= 2:
        final_step = all_weights[-1] - all_weights[-2]
        fs_norm = float(np.linalg.norm(final_step)) + 1e-12
        cos_pc1 = float(final_step @ pca.components_[0] / fs_norm)
        cos_pc2 = float(final_step @ pca.components_[1] / fs_norm)
        print(
            f"Final-step alignment  cos(Δθ, PC1)={cos_pc1:+.3f}  "
            f"cos(Δθ, PC2)={cos_pc2:+.3f}  "
            f"‖span‖²={cos_pc1 ** 2 + cos_pc2 ** 2:.3f}"
        )
    else:
        cos_pc1 = cos_pc2 = float("nan")

    # ---- Build env + inference factory (shared across bases) ------------- #
    env = _build_eval_env(args.grid * args.grid, morphology, n_pairs)
    inference_factory = _build_inference_factory(env, ppo_cfg)

    center_policy_tree = final_params[1]
    _, unravel_fn = ravel_pytree(center_policy_tree)
    # If we re-centered on a non-final snapshot, update the tree too.
    if not np.allclose(center_policy_flat, final_policy_flat):
        center_policy_tree = unravel_fn(jp.asarray(center_policy_flat))

    bases_to_run = (
        ["pca", "random"] if args.basis == "both" else [args.basis]
    )
    print(
        f"Running basis(es): {bases_to_run}  "
        f"filter_normalize={args.filter_normalize}  "
        f"seeds_per_cell={args.seeds_per_cell}"
    )

    # ---- Wave-analysis setup is shared across bases ---------------------- #
    groups = _group_joints(env.mj_model, morphology)
    print(
        f"Joint groups  spine={len(groups['spine'])}  "
        f"right={len(groups['right'])}  left={len(groups['left'])}"
    )
    ctrl_dt = float(env._config.ctrl_dt)
    action_scale = float(getattr(env._config, "action_scale", np.pi / 2.0))

    for basis_name in bases_to_run:
        print(f"\n==================  basis={basis_name}  ==================")
        basis = _build_basis(
            basis_name=basis_name,
            center_policy_tree=center_policy_tree,
            center_policy_flat=center_policy_flat,
            all_weights=all_weights,
            pca=pca,
            unravel_fn=unravel_fn,
            filter_normalize=args.filter_normalize,
            random_scale=args.random_scale,
            pc_range=args.pc_range,
            rng_seed=args.seed + 42,
        )
        print(
            f"  direction norms  ‖d1‖={basis['meta']['||d1||']:.3f}  "
            f"‖d2‖={basis['meta']['||d2||']:.3f}  "
            f"cos(d1,d2)={basis['meta']['cos(d1,d2)']:+.3f}  "
            f"‖θ_c‖={basis['meta']['||θ_c||']:.3f}"
        )
        print(f"  sweep half-widths  s1={basis['s1']:.4f}  s2={basis['s2']:.4f}")

        result = build_landscape(
            env=env,
            inference_factory=inference_factory,
            final_params=final_params,
            center_policy_flat=center_policy_flat,
            basis=basis,
            grid_n=args.grid,
            episode_length=args.episode_length,
            seed=args.seed,
            seeds_per_cell=args.seeds_per_cell,
        )
        A = result["A"]; B = result["B"]
        Z_disp = result["Z_disp_mean"]
        Z_disp_std = result["Z_disp_std"]
        Z_reward = result["Z_reward_mean"]
        Z_reward_std = result["Z_reward_std"]
        qpos_series = result["qpos_series"]
        action_series = result["action_series"]
        alphas = result["alphas"]; betas = result["betas"]

        # ---- Diagnostics: Hessian at center ------------------------------ #
        dalpha = float(alphas[1] - alphas[0]) if args.grid > 1 else 1.0
        dbeta = float(betas[1] - betas[0]) if args.grid > 1 else 1.0
        hess_disp = hessian_at_center(Z_disp, dalpha, dbeta)
        hess_reward = hessian_at_center(Z_reward, dalpha, dbeta)

        def _fmt_conv(v: float) -> str:
            return f"{v:+.3f}" if np.isfinite(v) else "n/a"

        print(
            f"  Hessian (disp)   shape={hess_disp['shape']}  "
            f"λ_max={hess_disp['lam_max']:+.3e}  "
            f"λ_min={hess_disp['lam_min']:+.3e}  "
            f"convexity={_fmt_conv(hess_disp['convexity'])}"
        )
        if args.reward:
            print(
                f"  Hessian (reward) shape={hess_reward['shape']}  "
                f"λ_max={hess_reward['lam_max']:+.3e}  "
                f"λ_min={hess_reward['lam_min']:+.3e}  "
                f"convexity={_fmt_conv(hess_reward['convexity'])}"
            )

        # ---- Compose subtitle shared by surface + 1D ---------------------- #
        if basis_name == "pca":
            evr_sub = (
                f"PC1={evr_full[0] * 100:.1f}% PC2={evr_full[1] * 100:.1f}% "
                f"(2PC cum={cumvar[1] * 100:.1f}%)  "
                f"cos(Δθ_K, PC1,PC2)=({cos_pc1:+.2f},{cos_pc2:+.2f})"
            )
        else:
            evr_sub = (
                f"random dirs  ‖d1‖/‖θ_c‖={basis['meta']['||d1||']/basis['meta']['||θ_c||']:.2f} "
                f"‖d2‖/‖θ_c‖={basis['meta']['||d2||']/basis['meta']['||θ_c||']:.2f} "
                f"cos(d1,d2)={basis['meta']['cos(d1,d2)']:+.2f}"
            )
        hess_sub = (
            f"H@0 [{hess_disp['shape']}]: λ_max={hess_disp['lam_max']:+.2e} "
            f"λ_min={hess_disp['lam_min']:+.2e} "
            f"conv={_fmt_conv(hess_disp['convexity'])}  "
            f"fnorm={args.filter_normalize}"
        )
        subtitle = evr_sub + "\n" + hess_sub

        # ---- Displacement landscape --------------------------------------- #
        landscape_path = run_dir / (
            f"landscape_ppo_{basis_name}_{args.grid}x{args.grid}.png"
        )
        plot_landscape(
            A=A, B=B, Z=Z_disp,
            output_path=landscape_path,
            label=center_label,
            xlabel=basis["label1"],
            ylabel=basis["label2"],
            zlabel="Final displacement (m)",
            title=f"PPO fitness landscape ({basis_name} slice)",
            subtitle=subtitle,
        )

        # ---- 1D cross-sections through the center ------------------------- #
        slice_path = run_dir / (
            f"landscape_ppo_{basis_name}_1d_{args.grid}x{args.grid}.png"
        )
        plot_1d_slices(
            alphas=alphas, betas=betas,
            Z_mean=Z_disp,
            Z_std=Z_disp_std if args.seeds_per_cell > 1 else None,
            xlabel=basis["label1"],
            ylabel="Final displacement (m)",
            output_path=slice_path,
            title=f"1D cross-sections through center ({basis_name})",
            subtitle=subtitle,
        )

        # ---- Reward landscape (optional) --------------------------------- #
        if args.reward:
            reward_path = run_dir / (
                f"landscape_ppo_{basis_name}_reward_{args.grid}x{args.grid}.png"
            )
            plot_landscape(
                A=A, B=B, Z=Z_reward,
                output_path=reward_path,
                label=center_label,
                xlabel=basis["label1"],
                ylabel=basis["label2"],
                zlabel="Mean per-step reward",
                title=f"PPO reward landscape ({basis_name} slice)",
                subtitle=subtitle,
            )

        # ---- Noise band heatmap (if multi-seed) -------------------------- #
        if args.seeds_per_cell > 1:
            std_path = run_dir / (
                f"landscape_ppo_{basis_name}_std_{args.grid}x{args.grid}.png"
            )
            plot_std_heatmap(
                A=A, B=B, Z_std=Z_disp_std,
                xlabel=basis["label1"],
                ylabel=basis["label2"],
                output_path=std_path,
                title=f"Displacement std over {args.seeds_per_cell} reset seeds "
                      f"({basis_name})",
            )

        # ---- Wave propagation analysis ----------------------------------- #
        if len(groups["right"]) >= 2 or len(groups["left"]) >= 2:
            Z_wave_flat = _wave_metric_landscape(qpos_series, groups, ctrl_dt)
            Z_wave = Z_wave_flat.reshape(args.grid, args.grid)
            wave_path = run_dir / (
                f"landscape_ppo_{basis_name}_wave_{args.grid}x{args.grid}.png"
            )
            plot_wave_landscape(
                A, B, Z_wave,
                xlabel=basis["label1"],
                ylabel=basis["label2"],
                output_path=wave_path,
            )

            qpos_grid = qpos_series.reshape(args.grid, args.grid, args.episode_length, -1)
            action_grid = action_series.reshape(args.grid, args.grid, args.episode_length, -1)
            picks = _select_picks(Z_disp, Z_wave, args.grid)
            picks_path = run_dir / (
                f"landscape_ppo_{basis_name}_wave_picks_{args.grid}x{args.grid}.png"
            )
            plot_wave_picks(
                picks=picks,
                qpos_series=qpos_grid,
                action_series=action_grid,
                groups=groups,
                ctrl_dt=ctrl_dt,
                A=A, B=B, Z_disp=Z_disp, Z_wave=Z_wave,
                output_path=picks_path,
                action_scale=action_scale,
            )
        else:
            print("Not enough joints on either side (<2) — skipping wave analysis.")

        # ---- Migration animation (only when PCA — trajectory lives there) #
        if args.animate and basis_name == "pca":
            rewards_per_snapshot = (
                np.array(history["reward"])
                if history is not None
                else np.full(len(snapshots), np.nan)
            )
            anim_path = run_dir / (
                f"landscape_ppo_{basis_name}_migration_{args.grid}x{args.grid}.gif"
            )
            animate_migration(
                A=A, B=B, Z=Z_disp,
                xlabel=basis["label1"],
                ylabel=basis["label2"],
                all_weights=all_weights,
                rewards_per_snapshot=rewards_per_snapshot,
                d1_flat=basis["d1_flat"],
                d2_flat=basis["d2_flat"],
                center_policy_flat=center_policy_flat,
                output_path=anim_path,
                fps=args.fps,
            )


if __name__ == "__main__":
    main()
