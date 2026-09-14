"""Visualizations for the morphology break-threshold study.

Produces four figures explaining the experimental setup:

  1. morph_geometry.png    — canonical hex layout with spin/arm indices
  2. perturbation_examples.png — what "arm +20 deg" and sigma=15 deg noise
                                 look like geometrically
  3. training_curve.png    — PPO training-side metrics (parsed from the log)
  4. sample_trajectories.png — 3D + top-down rollouts of the trained policy
                                 on canonical vs mildly vs severely perturbed
                                 morphs, so the drift-vs-perturbation regime
                                 is visible before the aggregate sweep lands.

Runs entirely on CPU to avoid contending with a sweep that may be running
on GPU. Reads the trained policy from morph_break_out/hover_policy.zip.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mba", str(Path(__file__).parent / "40_morph_break_analysis.py"),
)
mba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mba)

ROOT = Path(__file__).parent / "morph_break_out"
ROOT.mkdir(exist_ok=True)
TRAIN_LOG = Path("/tmp/morph_break_train2.log")
POLICY = ROOT / "hover_policy.zip"
# Set at __main__: OUT = ROOT / <mode>
OUT: Path = ROOT


# ---------------------------------------------------------------- geometry

def _propeller_xy(morph: dict) -> tuple[np.ndarray, list[str]]:
    locs = np.array([p["loc"] for p in morph["propellers"]], dtype=np.float32)
    spins = [p["dir"][3] if len(p["dir"]) > 3 else "?" for p in morph["propellers"]]
    return locs, spins


def _draw_hex(ax, morph: dict, title: str, ref_locs: np.ndarray | None = None):
    locs, spins = _propeller_xy(morph)
    # arms
    for i, (x, y, _) in enumerate(locs):
        ax.plot([0, x], [0, y], "k-", lw=1.5, alpha=0.6)
    # optional dashed reference (canonical positions)
    if ref_locs is not None:
        for x, y, _ in ref_locs:
            ax.plot([0, x], [0, y], "k--", lw=0.8, alpha=0.25)
    # propellers (color-coded by spin)
    for i, ((x, y, _), s) in enumerate(zip(locs, spins)):
        c = "tab:blue" if s == "ccw" else "tab:red"
        ax.scatter(x, y, s=250, c=c, edgecolors="black", zorder=3)
        ax.text(x * 1.20, y * 1.20, str(i), ha="center", va="center", fontsize=9)
    # core
    ax.scatter(0, 0, s=80, c="k", marker="s", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)


def plot_geometry():
    g = mba.canonical_genome()
    m = mba.genome_to_morph(g)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _draw_hex(ax, m,
              f"Canonical hex\nmass={m['mass']:.3f} kg  "
              f"twr={m['twr']:.2f}\nblue=CCW  red=CW  (indices 0..5)")
    fig.suptitle("Subject of the break-threshold study", fontsize=11)
    fig.tight_layout()
    p = OUT / "morph_geometry.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_perturbations(mode: str = "az"):
    g = mba.canonical_genome()
    canon = mba.genome_to_morph(g)
    canon_locs, _ = _propeller_xy(canon)

    vec = np.zeros(6, np.float32); vec[0] = math.radians(20)
    m_single = mba.genome_to_morph(mba.perturbed_genome(g, vec, mode=mode))
    rng = np.random.RandomState(1000)
    noise = rng.normal(0.0, math.radians(15.0), size=6).astype(np.float32)
    m_multi = mba.genome_to_morph(mba.perturbed_genome(g, noise, mode=mode))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    _draw_hex(axes[0], canon, "Canonical\n(reference)")
    if mode == "az":
        label_single = (f"Sweep A: arm 0 az +20 deg\n"
                        f"min_gap={mba.min_azimuth_gap_deg(m_single):.1f} deg")
        label_multi = (f"Sweep B: sigma=15 deg az noise\n"
                       f"offsets (deg) = {np.round(np.degrees(noise), 1)}\n"
                       f"min_gap={mba.min_azimuth_gap_deg(m_multi):.1f} deg")
    else:
        label_single = (f"Sweep A: arm 0 pitch +20 deg (tilt up)\n"
                        f"max |pitch|={mba.max_pitch_deg(m_single):.1f} deg\n"
                        f"top-down view: horizontal projection shrinks")
        label_multi = (f"Sweep B: sigma=15 deg pitch noise\n"
                       f"offsets (deg) = {np.round(np.degrees(noise), 1)}\n"
                       f"max |pitch|={mba.max_pitch_deg(m_multi):.1f} deg")
    _draw_hex(axes[1], m_single, label_single, ref_locs=canon_locs)
    _draw_hex(axes[2], m_multi, label_multi, ref_locs=canon_locs)
    fig.suptitle(f"What each sweep perturbs — mode={mode}   "
                 f"(dashed = canonical arm positions in XY)", fontsize=11)
    fig.tight_layout()
    p = OUT / "perturbation_examples.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- training

def _parse_log(path: Path) -> dict[str, list[float]]:
    """Return dict keyed by SB3 metric name, values over successive rollouts."""
    if not path.exists():
        return {}
    text = path.read_text()
    keys = ("total_timesteps", "explained_variance", "value_loss", "std",
            "approx_kl", "entropy_loss", "clip_fraction")
    out: dict[str, list[float]] = {k: [] for k in keys}
    for k in keys:
        for m in re.finditer(rf"\|\s*{k}\s*\|\s*([\-0-9\.e]+)\s*\|", text):
            try:
                out[k].append(float(m.group(1)))
            except ValueError:
                pass
    return out


def plot_training():
    d = _parse_log(TRAIN_LOG)
    if not d.get("total_timesteps"):
        print(f"  skip training plot: {TRAIN_LOG} missing or empty")
        return
    ts = np.array(d["total_timesteps"], dtype=float) / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    def _pair(ax, key, ylabel, log=False):
        y = np.array(d.get(key, []), dtype=float)
        n = min(len(ts), len(y))
        ax.plot(ts[:n], y[:n], lw=1.2)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    _pair(axes[0, 0], "explained_variance",
          "explained_variance\n(value fn quality)")
    _pair(axes[0, 1], "std",
          "policy action std\n(exploration)")
    _pair(axes[1, 0], "value_loss",
          "value_loss  (log)", log=True)
    _pair(axes[1, 1], "approx_kl",
          "approx_kl per update")

    for ax in axes[-1]:
        ax.set_xlabel("training steps (millions)")
    fig.suptitle("PPO hover specialist — training convergence "
                 "(20M steps on GPU, canonical hex)", fontsize=11)
    fig.tight_layout()
    p = OUT / "training_curve.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- trajectories

def _rollout(model, morph: dict, seed: int = 42) -> np.ndarray:
    """Return (T, 3) NED positions from a single deterministic rollout."""
    env = mba.make_env(morph, num_envs=1, seed=seed,
                       frozen_features=None, device="cpu")
    obs = env.reset()
    positions = []
    for t in range(mba.EPISODE_STEPS):
        actions, _ = model.predict(obs, deterministic=True)
        env.step_async(actions)
        obs, r, dones, infos = env.step_wait()
        positions.append(env.world_states[0, 0:3].cpu().numpy().copy())
        if bool(dones[0]):
            break
    return np.array(positions)


def plot_trajectories(mode: str = "az"):
    if not POLICY.exists():
        print(f"  skip trajectories: {POLICY} missing")
        return
    from stable_baselines3 import PPO
    model = PPO.load(str(POLICY), device="cpu")

    g = mba.canonical_genome()
    axis = "az" if mode == "az" else "pitch"
    scenarios = [
        ("canonical", np.zeros(6, np.float32), "tab:green"),
        (f"arm 0 {axis} +10 deg", _single_offset(0, 10), "tab:orange"),
        (f"arm 0 {axis} +30 deg", _single_offset(0, 30), "tab:red"),
    ]
    trajs = []
    for name, offsets, color in scenarios:
        m = mba.genome_to_morph(mba.perturbed_genome(g, offsets, mode=mode))
        pos = _rollout(model, m)
        trajs.append((name, pos, color))
        print(f"    rolled out '{name}': {len(pos)} steps, "
              f"final drift={np.linalg.norm(pos[-1] - np.array([0,0,-1.5])):.3f} m")

    target = np.array([0.0, 0.0, -1.5])
    fig = plt.figure(figsize=(14, 5))

    # Top-down XY
    ax1 = fig.add_subplot(1, 3, 1)
    for name, pos, color in trajs:
        ax1.plot(pos[:, 0], pos[:, 1], color=color, lw=1.4, label=name)
        ax1.scatter(pos[0, 0], pos[0, 1], color=color, marker="o", s=40)
        ax1.scatter(pos[-1, 0], pos[-1, 1], color=color, marker="X", s=60)
    ax1.scatter(0, 0, color="black", marker="*", s=120, label="hover target (xy)")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m, NED)")
    ax1.set_ylabel("y (m, NED)")
    ax1.set_title("Top-down (o=start, X=end)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="best")

    # Altitude over time
    ax2 = fig.add_subplot(1, 3, 2)
    for name, pos, color in trajs:
        t = np.arange(len(pos)) * 0.01
        ax2.plot(t, -pos[:, 2], color=color, lw=1.4, label=name)
    ax2.axhline(1.5, color="black", ls="--", lw=0.8, label="hover altitude")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("altitude above ground (m)")
    ax2.set_title("Altitude over time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Drift-from-target
    ax3 = fig.add_subplot(1, 3, 3)
    for name, pos, color in trajs:
        t = np.arange(len(pos)) * 0.01
        d = np.linalg.norm(pos - target, axis=1)
        ax3.plot(t, d, color=color, lw=1.4, label=name)
    ax3.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 m success threshold")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("|position - target| (m)")
    ax3.set_title("Drift over time")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle("How the trained specialist responds to arm-angle perturbation",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / "sample_trajectories.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def _single_offset(arm: int, deg: float) -> np.ndarray:
    v = np.zeros(6, np.float32)
    v[arm] = math.radians(deg)
    return v


# ---------------------------------------------------------------- main

def _read_csv(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def plot_sweep_results():
    rows_a = _read_csv(OUT / "single_arm.csv")
    rows_b = _read_csv(OUT / "all_arm.csv")
    if not rows_a or not rows_b:
        print("  skip results plot: sweep CSVs missing")
        return

    arms = sorted({int(r["arm"]) for r in rows_a})
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Sweep A: single-arm, success rate + drift ---
    ax = axes[0, 0]
    for arm in arms:
        sub = sorted((r for r in rows_a if int(r["arm"]) == arm),
                     key=lambda r: r["offset_deg"])
        ax.plot([r["offset_deg"] for r in sub],
                [r["success_rate"] for r in sub],
                marker="o", ms=3, color=cmap(arm), label=f"arm {arm}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("single-arm offset (deg)")
    ax.set_ylabel("success rate  (drift < 0.5 m)")
    ax.set_title("Sweep A: hard-metric — success rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for arm in arms:
        sub = sorted((r for r in rows_a if int(r["arm"]) == arm),
                     key=lambda r: r["offset_deg"])
        drifts = [r["mean_final_drift"] if r["mean_final_drift"] == r["mean_final_drift"] else np.nan
                  for r in sub]
        ax.plot([r["offset_deg"] for r in sub], drifts,
                marker="o", ms=3, color=cmap(arm), label=f"arm {arm}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 m gate")
    ax.set_xlabel("single-arm offset (deg)")
    ax.set_ylabel("mean final drift (m)  [survivors only]")
    ax.set_title("Sweep A: soft-metric — actual drift\n"
                 "(smooth degradation invisible in success rate)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("symlog", linthresh=1.0)

    # --- Sweep B: all-arm sigma, aggregate stats ---
    sigmas = sorted({r["sigma_deg"] for r in rows_b})
    def _stat(sig, key):
        vals = [r[key] for r in rows_b if r["sigma_deg"] == sig]
        vals = [v for v in vals if v == v]  # drop nan
        return (np.mean(vals) if vals else np.nan,
                np.std(vals) if vals else np.nan)

    ax = axes[1, 0]
    means = [_stat(s, "success_rate")[0] for s in sigmas]
    stds = [_stat(s, "success_rate")[1] for s in sigmas]
    surv = [_stat(s, "survival_rate")[0] for s in sigmas]
    ax.errorbar(sigmas, means, yerr=stds, marker="o", ms=4, capsize=3,
                label="success (drift<0.5m)", color="tab:blue")
    ax.plot(sigmas, surv, marker="s", ms=4, ls="--",
            label="survival (600 steps)", color="tab:green")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("all-arm noise sigma (deg)")
    ax.set_ylabel("rate")
    ax.set_title("Sweep B: success vs survival\n"
                 "(gap reveals failures are drift, not crashes)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    drift_mean = [_stat(s, "mean_final_drift")[0] for s in sigmas]
    drift_std = [_stat(s, "mean_final_drift")[1] for s in sigmas]
    ax.errorbar(sigmas, drift_mean, yerr=drift_std, marker="o", ms=4, capsize=3,
                color="tab:red")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 m gate")
    ax.set_xlabel("all-arm noise sigma (deg)")
    ax.set_ylabel("mean final drift (m)")
    ax.set_title("Sweep B: drift grows smoothly with noise")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(fontsize=8)

    fig.suptitle("PPO hover specialist — response to arm-angle perturbations\n"
                 "(20M steps, canonical hex, 16 rollouts per morph)", fontsize=12)
    fig.tight_layout()
    p = OUT / "sweep_results_detailed.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["az", "pitch"], default="az")
    args = p.parse_args()
    OUT = ROOT / args.mode
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[viz] mode={args.mode}  out={OUT}")
    print("[viz] geometry ..."); plot_geometry()
    print("[viz] perturbation examples ..."); plot_perturbations(mode=args.mode)
    print("[viz] training curve ..."); plot_training()
    print("[viz] sample trajectories ..."); plot_trajectories(mode=args.mode)
    print("[viz] sweep results ..."); plot_sweep_results()
    print("Done. See:", OUT)
