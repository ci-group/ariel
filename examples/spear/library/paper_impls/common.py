"""Shared helpers for the paper implementations.

Baseline loading, the standard metric, timestamped output directories, and
per-task episode aggregators used across p1–p6. Everything in here wraps
existing code (`ResidualDroneEnv`, `MorphRotatingVecEnv`, evaluator utilities
from `37_train_residual_mtrl.py`); nothing here duplicates it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

LIBRARY_ROOT = Path(__file__).parents[4]   # ariel/
SCRIPTS_DIR = Path(__file__).parents[1]    # examples/spear/library/
OUT_ROOT = LIBRARY_ROOT / "__data__" / "paper_impls"


def _load_script_module(name: str, file_stem: str):
    """Import a sibling numbered script (e.g. `37_train_residual_mtrl.py`).

    The file starts with a digit, so a normal `import` won't work.
    """
    if name in sys.modules:
        return sys.modules[name]
    path = SCRIPTS_DIR / f"{file_stem}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Ensure the library dir is on sys.path so the script's own imports
    # (`envs.residual_drone_env`, `hex_sampler`, ...) resolve.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec.loader.exec_module(mod)
    return mod


def load_37():
    return _load_script_module("t37", "37_train_residual_mtrl")


def load_40():
    return _load_script_module("t40", "40_morph_break_analysis")


def load_42a():
    return _load_script_module("t42a", "42a_draw_trajectory")


def load_42b():
    return _load_script_module("t42b", "42b_train_blueprint_traj")


# ─────────────────────────────────────────────────────────────────────────────
# Standard metric (PAPER_IMPL_PLAN §"ground rules"):
#   metric = (hover + 2 · (figure8 + slalom + shuttle-run + circle)) / 9
# Trajectory tasks weighted 2× because they carry the residual's real workload;
# hover is a stability sanity check for which the prior already solves most of
# the problem. This mirrors autoresearch_program.md.
# ─────────────────────────────────────────────────────────────────────────────

TASK_NAMES = ("hover", "figure8", "slalom", "shuttle-run", "circle")


def standard_metric(mean_reward_per_task: dict[str, float]) -> float:
    """Compute (hover + 2·traj) / 9 from mean episode rewards."""
    hover = float(mean_reward_per_task.get("hover", 0.0))
    traj = sum(
        float(mean_reward_per_task.get(t, 0.0))
        for t in ("figure8", "slalom", "shuttle-run", "circle")
    )
    return (hover + 2.0 * traj) / 9.0


def summarize_eval(ep_r: dict[str, list[float]]) -> tuple[dict[str, float], float]:
    means = {t: (float(np.mean(ep_r[t])) if ep_r[t] else 0.0) for t in TASK_NAMES}
    return means, standard_metric(means)


# ─────────────────────────────────────────────────────────────────────────────
# Output plumbing
# ─────────────────────────────────────────────────────────────────────────────

def new_out_dir(paper_tag: str, label: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sub = f"{stamp}_{label}" if label else stamp
    d = OUT_ROOT / paper_tag / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_config(out_dir: Path, cfg: dict[str, Any]) -> None:
    def _cast(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()
        return v
    (out_dir / "config.json").write_text(
        json.dumps({k: _cast(v) for k, v in cfg.items()}, indent=2),
    )


def dump_results(out_dir: Path, results: dict[str, Any]) -> None:
    def _cast(v):
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: _cast(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_cast(x) for x in v]
        return v
    (out_dir / "results.json").write_text(json.dumps(
        {k: _cast(v) for k, v in results.items()}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers for α-tag experiments (P4) and paired baseline runs (P1/P5/P6).
# Keep these tiny — anything larger belongs in the paper file itself.
# ─────────────────────────────────────────────────────────────────────────────

def print_metric_line(prefix: str, means: dict[str, float], metric: float):
    line = ", ".join(f"{t}={means[t]:+7.2f}" for t in TASK_NAMES)
    print(f"[{prefix}] metric={metric:+7.3f}  |  {line}", flush=True)
