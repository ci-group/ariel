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
    <run-dir>/landscape_ppo_pca_<grid>x<grid>.png
    <run-dir>/landscape_ppo_scree.png
    <run-dir>/landscape_ppo_reward.png
    <run-dir>/landscape_ppo_migration_<grid>x<grid>.gif  (if --animate)

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
def _build_eval_env(num_envs: int) -> UndirectedLocomotionInsect:
    """Env sized for ``num_envs`` concurrent rollouts (used with jax.vmap)."""
    cfg = default_config()
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


def build_landscape(
    env: UndirectedLocomotionInsect,
    inference_factory: Any,
    final_params: tuple,
    all_weights: np.ndarray,
    center_policy_flat: np.ndarray,
    pca: PCA,
    pc_range: float,
    grid_n: int,
    episode_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a grid of perturbed policies around ``center_policy_flat``.

    Keeps normalizer + value params fixed at the final-training values and
    only perturbs policy_params along the top two PCs. Returns ``(A, B, Z)``
    meshgrids in PC coordinates and final-displacement meters.
    """
    final_normalizer, final_policy, final_value = final_params
    _, unravel_fn = ravel_pytree(final_policy)

    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    # Scale grid by the stdev of the training trajectory along each PC.
    projected = pca.transform(all_weights)
    std1 = float(projected[:, 0].std())
    std2 = float(projected[:, 1].std())

    alphas = np.linspace(-pc_range * std1, pc_range * std1, grid_n)
    betas = np.linspace(-pc_range * std2, pc_range * std2, grid_n)

    grid_flat = np.stack(
        [center_policy_flat + a * pc1 + b * pc2 for a in alphas for b in betas]
    ).astype(np.float32)

    reset_rng = jax.random.PRNGKey(seed + 100)
    scan_rng = jax.random.PRNGKey(seed + 101)

    def _single_rollout(policy_flat: jax.Array) -> jax.Array:
        policy_params = unravel_fn(policy_flat)
        params = (final_normalizer, policy_params, final_value)
        policy = inference_factory(params, deterministic=True)
        state = env.reset(reset_rng)

        def body(carry, _):
            state, rng = carry
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
            return (env.step(state, action), rng), None

        (final_state, _), _ = jax.lax.scan(
            body, (state, scan_rng), None, length=episode_length
        )
        disp = jp.linalg.norm(final_state.data.qpos[:2])
        return jp.where(jp.isnan(disp), 0.0, disp)

    batched = jax.jit(jax.vmap(_single_rollout))

    print(
        f"Evaluating {grid_n * grid_n} grid points on {jax.default_backend()}... "
        "(first call compiles the batched rollout)"
    )
    t0 = time.monotonic()
    displacements = np.asarray(batched(jp.asarray(grid_flat)))
    print(f"  landscape eval done in {time.monotonic() - t0:.1f} s")

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    Z = displacements.reshape(grid_n, grid_n)
    return A, B, Z


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
    evr: np.ndarray,
    center_policy_flat: np.ndarray,
    pca: PCA,
    output_path: Path,
    label: str,
) -> None:
    center_proj = pca.transform(center_policy_flat.reshape(1, -1))

    fig = plt.figure(figsize=(14, 6))

    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(A, B, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, label="Final displacement (m)")
    ax3d.scatter(
        center_proj[0, 0], center_proj[0, 1], Z.max(),
        color="red", s=60, zorder=5, label=label,
    )
    ax3d.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax3d.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    ax3d.set_zlabel("Final displacement (m)")
    ax3d.set_title("PPO fitness landscape (PCA slice)")
    ax3d.legend()

    ax2d = fig.add_subplot(122)
    hm = ax2d.contourf(A, B, Z, levels=30, cmap=cm.viridis)
    fig.colorbar(hm, ax=ax2d, label="Final displacement (m)")
    ax2d.scatter(
        center_proj[0, 0], center_proj[0, 1],
        color="red", s=60, zorder=5, label=label,
    )
    ax2d.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax2d.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    ax2d.set_title("PPO fitness landscape (top view)")
    ax2d.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Landscape saved to {output_path}")


def animate_migration(
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    evr: np.ndarray,
    all_weights: np.ndarray,
    rewards_per_snapshot: np.ndarray,
    pca: PCA,
    output_path: Path,
    fps: int,
) -> None:
    """Snapshot-by-snapshot animation of the policy drifting across the landscape."""
    projected = pca.transform(all_weights)
    xs = projected[:, 0]
    ys = projected[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(A, B, Z, levels=30, cmap=cm.viridis)
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(Z)
    fig.colorbar(mappable, ax=ax, label="Final displacement (m)")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")

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
        plot_reward(history, run_dir / "landscape_ppo_reward.png")

    pca = PCA(n_components=2).fit(all_weights)

    # ---- Build env, inference factory, landscape ------------------------- #
    env = _build_eval_env(args.grid * args.grid)
    inference_factory = _build_inference_factory(env, ppo_cfg)

    A, B, Z = build_landscape(
        env=env,
        inference_factory=inference_factory,
        final_params=final_params,
        all_weights=all_weights,
        center_policy_flat=center_policy_flat,
        pca=pca,
        pc_range=args.pc_range,
        grid_n=args.grid,
        episode_length=args.episode_length,
        seed=args.seed,
    )

    landscape_path = run_dir / f"landscape_ppo_pca_{args.grid}x{args.grid}.png"
    plot_landscape(
        A=A, B=B, Z=Z,
        evr=pca.explained_variance_ratio_,
        center_policy_flat=center_policy_flat,
        pca=pca,
        output_path=landscape_path,
        label=center_label,
    )

    if args.animate:
        rewards_per_snapshot = (
            np.array(history["reward"])
            if history is not None
            else np.full(len(snapshots), np.nan)
        )
        anim_path = run_dir / f"landscape_ppo_migration_{args.grid}x{args.grid}.gif"
        animate_migration(
            A=A, B=B, Z=Z,
            evr=pca.explained_variance_ratio_,
            all_weights=all_weights,
            rewards_per_snapshot=rewards_per_snapshot,
            pca=pca,
            output_path=anim_path,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
