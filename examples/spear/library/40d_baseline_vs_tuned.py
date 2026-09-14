"""Compare baseline vs tuned morph-break sweeps: side-by-side plots per axis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "morph_break_out"


def load(mode: str, tuned: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tuned:
        base = ROOT / "tuned" / mode
    else:
        base = ROOT if mode == "az" else ROOT / "pitch"
    return pd.read_csv(base / "single_arm.csv"), pd.read_csv(base / "all_arm.csv")


def sym_single(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("offset_deg", as_index=False).agg(
        success_rate=("success_rate", "mean"),
        mean_final_drift=("mean_final_drift", "mean"),
    )


def sigma_agg(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("sigma_deg", as_index=False).agg(
        success_rate=("success_rate", "mean"),
        mean_final_drift=("mean_final_drift", "mean"),
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, mode in enumerate(["az", "pitch"]):
        base_single, base_all = load(mode, tuned=False)
        tune_single, tune_all = load(mode, tuned=True)

        bs, ts = sym_single(base_single), sym_single(tune_single)
        ba, ta = sigma_agg(base_all), sigma_agg(tune_all)

        ax = axes[0, col]
        ax.plot(bs["offset_deg"], bs["success_rate"], "o-", color="tab:red",
                label="baseline", linewidth=2)
        ax.plot(ts["offset_deg"], ts["success_rate"], "s-", color="tab:green",
                label="tuned (shape+ent+vfclip)", linewidth=2)
        ax.set_xlabel(f"single-arm {mode} offset (deg)")
        ax.set_ylabel("mean success rate")
        ax.set_title(f"Single-arm {mode.upper()} perturbation")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.8, color="grey", linestyle=":", alpha=0.5, label="80% gate")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(ba["sigma_deg"], ba["success_rate"], "o-", color="tab:red",
                label="baseline", linewidth=2)
        ax.plot(ta["sigma_deg"], ta["success_rate"], "s-", color="tab:green",
                label="tuned", linewidth=2)
        ax.set_xlabel(f"all-arm {mode} σ (deg)")
        ax.set_ylabel("mean success rate")
        ax.set_title(f"All-arm {mode.upper()} σ sweep (8 draws)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.8, color="grey", linestyle=":", alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Morphology break: baseline vs tuned specialist\n"
                 "tuning = quadratic centering (c=0.1) + ent_coef=0.005 + clip_range_vf=0.2",
                 fontsize=13)
    fig.tight_layout()
    out = ROOT / "baseline_vs_tuned.png"
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
