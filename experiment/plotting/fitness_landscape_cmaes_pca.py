# Standard libraries
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
import torch
from torch import nn
import mujoco

from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.tasks.gait_learning import xy_displacement

# ============================================================================ #
#                          Command-line arguments                              #
# ============================================================================ #
import argparse

parser = argparse.ArgumentParser(description="PCA fitness landscape visualiser")
parser.add_argument(
    "--log",
    type=Path,
    required=True,
    help="Path to search_log.npz produced by undirected_locomotion.py",
)
parser.add_argument(
    "--morphology",
    type=str,
    default="centipede",
    choices=["spider", "centipede", "gecko"],
    help="Morphology used during the run (must match the log)",
)
parser.add_argument("--grid", type=int, default=25, help="Grid resolution per axis")
parser.add_argument(
    "--range",
    type=float,
    default=2.0,
    dest="pc_range",
    help="How many standard deviations along each PC to sweep",
)
parser.add_argument("--dur", type=int, default=10, help="Evaluation duration (seconds)")
parser.add_argument(
    "--workers",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Parallel worker processes",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--animate",
    action="store_true",
    help="Produce an animated GIF showing the best-solution dot migrating over generations",
)
parser.add_argument(
    "--fps",
    type=int,
    default=10,
    help="Frames per second for the animation",
)
args = parser.parse_args()

DURATION = args.dur
NUM_WORKERS = max(1, args.workers)
BASE_SEED = int(args.seed)

# ============================================================================ #
#                         Network (mirrors undirected_locomotion)              #
# ============================================================================ #

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ELU()
        self.output_act = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(x, dtype=torch.float32)
        t = self.hidden_act(self.fc1(t))
        t = self.hidden_act(self.fc2(t))
        t = self.output_act(self.fc3(t)) * (torch.pi / 2)
        return t.numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor) -> None:
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address: address + n], device=d.device)
        address += n
    if address != len(vector):
        raise IndexError("Parameter vector length mismatch")


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_worker(base_seed: int) -> None:
    torch.set_num_threads(1)
    worker_seed = (base_seed + os.getpid()) % (2**32 - 1)
    _seed_everything(worker_seed)


def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    duration: int,
    control_step_freq: int = 50,
) -> dict[str, Any]:
    current_action = np.zeros(model.nu)
    last_pos = data.qpos[0:3].copy()
    total_path_length = 0.0
    step = 0

    while data.time < duration:
        if step % control_step_freq == 0:
            robot_state = get_robot_state(data)
            phase = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]
            state = np.concatenate([robot_state, phase]).astype(np.float32)
            current_action = network.forward(state)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        current_pos = data.qpos[0:3].copy()
        total_path_length += float(np.linalg.norm(current_pos - last_pos))
        last_pos = current_pos

    final_xy = (float(data.qpos[0]), float(data.qpos[1]))
    return {
        "displacement": xy_displacement((0.0, 0.0), final_xy),
        "path_length": total_path_length,
    }


# ============================================================================ #
#                    Process-local simulation context                          #
# ============================================================================ #

_landscape_ctx: dict[str, Any] | None = None


def _load_body():
    if args.morphology == "spider":
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
        return spider()
    if args.morphology == "centipede":
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.centipede import body_centipede
        return body_centipede()
    if args.morphology == "gecko":
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
        return gecko()
    raise ValueError(f"Unknown morphology: {args.morphology}")


def _build_context() -> dict[str, Any]:
    world = SimpleFlatWorld()
    body = _load_body()
    world.spawn(body.spec, position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    num_joints = model.nq - 7
    input_dim = 3 + num_joints + 2
    network = Network(input_size=input_dim, output_size=model.nu, hidden_size=32)
    return {"model": model, "data": data, "network": network, "input_dim": input_dim}


def _get_context() -> dict[str, Any]:
    global _landscape_ctx
    if _landscape_ctx is None:
        _landscape_ctx = _build_context()
    return _landscape_ctx


def _evaluate_weights(weights: np.ndarray) -> float:
    ctx = _get_context()
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    network: Network = ctx["network"]
    fill_parameters(network, torch.as_tensor(weights, dtype=torch.float32))
    mujoco.mj_resetData(model, data)
    return run_simulation(model, data, network, DURATION)["displacement"]


# ============================================================================ #
#                              PCA landscape                                   #
# ============================================================================ #

def build_landscape(
    all_weights: np.ndarray,
    best_weights: np.ndarray,
    pca: PCA,
    pc_range: float,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]

    projected = pca.transform(all_weights)
    std1, std2 = projected[:, 0].std(), projected[:, 1].std()

    alphas = np.linspace(-pc_range * std1, pc_range * std1, grid_n)
    betas  = np.linspace(-pc_range * std2, pc_range * std2, grid_n)

    grid_weights = [
        best_weights + a * pc1 + b * pc2
        for a in alphas
        for b in betas
    ]

    print(f"Evaluating {len(grid_weights)} grid points with {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        displacements = list(executor.map(_evaluate_weights, grid_weights))

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    Z = np.array(displacements).reshape(grid_n, grid_n)
    return A, B, Z


def plot_landscape(
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    evr: np.ndarray,
    best_weights: np.ndarray,
    pca: PCA,
    output_path: Path,
) -> None:
    best_proj = pca.transform(best_weights.reshape(1, -1))

    fig = plt.figure(figsize=(14, 6))

    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(A, B, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, label="Displacement (m)")
    ax3d.scatter(
        best_proj[0, 0], best_proj[0, 1], Z.max(),
        color="red", s=60, zorder=5, label="Best solution",
    )
    ax3d.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax3d.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax3d.set_zlabel("Displacement (m)")
    ax3d.set_title("Fitness landscape (PCA slice)")
    ax3d.legend()

    ax2d = fig.add_subplot(122)
    hm = ax2d.contourf(A, B, Z, levels=30, cmap=cm.viridis)
    fig.colorbar(hm, ax=ax2d, label="Displacement (m)")
    ax2d.scatter(
        best_proj[0, 0], best_proj[0, 1],
        color="red", s=60, zorder=5, label="Best solution",
    )
    ax2d.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax2d.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax2d.set_title("Fitness landscape (top view)")
    ax2d.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.show()


def animate_migration(
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    evr: np.ndarray,
    all_weights: np.ndarray,
    all_fitnesses: np.ndarray,
    pop_size: int,
    pca: PCA,
    output_path: Path,
    fps: int,
) -> None:
    """Animate the cumulative-best solution moving across the PCA landscape."""
    n_gens = len(all_weights) // pop_size

    # Compute the cumulative best-so-far position for each generation.
    best_proj_per_gen: list[tuple[float, float]] = []
    best_fit_so_far = np.inf
    best_w_so_far = all_weights[0]

    for g in range(n_gens):
        gen_fits = all_fitnesses[g * pop_size: (g + 1) * pop_size]
        gen_weights = all_weights[g * pop_size: (g + 1) * pop_size]
        idx = int(np.argmin(gen_fits))
        if gen_fits[idx] < best_fit_so_far:
            best_fit_so_far = gen_fits[idx]
            best_w_so_far = gen_weights[idx]
        proj = pca.transform(best_w_so_far.reshape(1, -1))
        best_proj_per_gen.append((float(proj[0, 0]), float(proj[0, 1])))

    xs = [p[0] for p in best_proj_per_gen]
    ys = [p[1] for p in best_proj_per_gen]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(A, B, Z, levels=30, cmap=cm.viridis)
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(Z)
    fig.colorbar(mappable, ax=ax, label="Displacement (m)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")

    trail_line, = ax.plot([], [], color="white", linewidth=1, alpha=0.6)
    dot = ax.scatter([], [], color="red", s=80, zorder=6)
    title = ax.set_title("")

    def init():
        trail_line.set_data([], [])
        dot.set_offsets(np.empty((0, 2)))
        return trail_line, dot

    def update(frame: int):
        trail_line.set_data(xs[:frame + 1], ys[:frame + 1])
        dot.set_offsets([[xs[frame], ys[frame]]])
        title.set_text(
            f"Gen {frame + 1}/{n_gens}  |  "
            f"best displacement: {-all_fitnesses[:( frame + 1) * pop_size].min():.3f} m"
        )
        return trail_line, dot, title

    anim = FuncAnimation(
        fig, update, frames=n_gens, init_func=init, blit=True, interval=1000 // fps
    )
    anim.save(output_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved to {output_path}")
    plt.close(fig)


# ============================================================================ #
#                                    Main                                      #
# ============================================================================ #

if __name__ == "__main__":
    _seed_everything(BASE_SEED)
    mujoco.set_mjcb_control(None)

    log = np.load(args.log)
    all_weights: np.ndarray = log["weights"]       # (N, n_params)
    all_fitnesses: np.ndarray = log["fitnesses"]   # (N,) negated displacements
    pop_size: int = int(log["pop_size"]) if "pop_size" in log else 30
    sigmas: np.ndarray | None = log["sigmas"] if "sigmas" in log else None

    print(f"Loaded {len(all_weights)} candidates ({len(all_weights) // pop_size} gens) from {args.log}")
    print(f"Best displacement in log: {-float(all_fitnesses.min()):.4f} m")

    best_weights = all_weights[int(np.argmin(all_fitnesses))]
    saved_best = args.log.parent / "best_weights.npy"
    if saved_best.exists():
        best_weights = np.load(saved_best)
        print(f"Using best_weights.npy from {saved_best}")

    # Full PCA to diagnose effective dimensionality before fitting the 2-PC version.
    pca_full = PCA().fit(all_weights)
    evr_full = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr_full)
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"PCs needed for 95% variance: {n95}  |  99%: {n99}  |  total PCs: {len(evr_full)}")

    scree_path = args.log.parent / "scree_plot.png"
    fig_s, (ax_ind, ax_cum) = plt.subplots(1, 2, figsize=(12, 4))

    ax_ind.bar(range(1, min(31, len(evr_full) + 1)), evr_full[:30] * 100, color="steelblue")
    ax_ind.set_xlabel("Principal component")
    ax_ind.set_ylabel("Explained variance (%)")
    ax_ind.set_title("Individual explained variance (top 30 PCs)")
    ax_ind.axvline(2.5, color="red", linestyle="--", label="2-PC cutoff")
    ax_ind.legend()

    ax_cum.plot(range(1, len(evr_full) + 1), cumvar * 100, color="steelblue")
    ax_cum.axhline(95, color="orange", linestyle="--", label="95%")
    ax_cum.axhline(99, color="red", linestyle="--", label="99%")
    ax_cum.axvline(2, color="grey", linestyle=":", label="2 PCs shown in landscape")
    ax_cum.set_xlabel("Number of principal components")
    ax_cum.set_ylabel("Cumulative explained variance (%)")
    ax_cum.set_title(f"Cumulative variance  (95% @ PC{n95}, 99% @ PC{n99})")
    ax_cum.legend()

    plt.tight_layout()
    plt.savefig(scree_path, dpi=150)
    print(f"Scree plot saved to {scree_path}")
    plt.close(fig_s)

    if sigmas is not None:
        n_gens = len(sigmas)
        generations = np.arange(1, n_gens + 1)

        # Per-generation best displacement (cumulative best).
        best_per_gen = np.array([
            -float(all_fitnesses[: (g + 1) * pop_size].min())
            for g in range(n_gens)
        ])

        sigma_path = args.log.parent / "sigma_plot.png"
        fig_sig, (ax_sig, ax_fit) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax_sig.plot(generations, sigmas, color="steelblue", linewidth=1.5)
        ax_sig.set_ylabel("CMA-ES sigma (step size)")
        ax_sig.set_title("CMA-ES step size over generations")
        ax_sig.set_yscale("log")
        ax_sig.grid(True, alpha=0.3)

        ax_fit.plot(generations, best_per_gen, color="darkorange", linewidth=1.5)
        ax_fit.set_ylabel("Best displacement (m)")
        ax_fit.set_xlabel("Generation")
        ax_fit.set_title("Cumulative best fitness over generations")
        ax_fit.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(sigma_path, dpi=150)
        print(f"Sigma plot saved to {sigma_path}")
        plt.close(fig_sig)
    else:
        print("No sigma data in log — re-run evolution to capture it.")

    pca = PCA(n_components=2)
    pca.fit(all_weights)
    print(
        f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}%"
    )

    A, B, Z = build_landscape(
        all_weights=all_weights,
        best_weights=best_weights,
        pca=pca,
        pc_range=args.pc_range,
        grid_n=args.grid,
    )

    output_path = args.log.parent / f"landscape_pca_{args.grid}x{args.grid}.png"
    plot_landscape(A, B, Z, pca.explained_variance_ratio_, best_weights, pca, output_path)

    if args.animate:
        anim_path = args.log.parent / f"landscape_migration_{args.grid}x{args.grid}.gif"
        animate_migration(
            A=A, B=B, Z=Z,
            evr=pca.explained_variance_ratio_,
            all_weights=all_weights,
            all_fitnesses=all_fitnesses,
            pop_size=pop_size,
            pca=pca,
            output_path=anim_path,
            fps=args.fps,
        )
