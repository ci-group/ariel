"""Can a linear policy do better than PPO's linear policy?  (DAgger distillation)

Question (about PPO, not about morphology): the linear PPO actor plateaus at
~2.9 targets/episode while the w256 d3 actor reaches ~4.5. Is that gap

  (a) representational — no linear policy can do much better, or
  (b) an optimisation failure of PPO — better linear policies exist, PPO
      just doesn't find them?

Test: distil a trained w256 d3 PPO policy (the teacher) into the SAME policy
class PPO's linear actor uses, a = tanh(W·obs + b) on the actor's "state"
observation (brax's observation normaliser is affine, so it is absorbed in
W, b), with DAgger:

    iter 0  : teacher drives, record (state obs, teacher action)
    iter i>0: current student drives, teacher labels the states it visits
    each    : refit the student on all data so far (aggregate)

The student is fitted by least squares in pre-tanh space (init) and then by
Adam on the action-space MSE  ‖tanh(W·x + b) − a_teacher‖².

Control: the same pipeline distils into a w256 d3 MLP student. If the control
does not recover the teacher's score, the pipeline itself loses performance
and the linear result can't be interpreted.

All policies are evaluated in the same harness: N parallel episodes of
episode_length steps on the raw env, deterministic actions (tanh of the
mean), metric = targets reached per episode, counting only steps before the
robot falls (done). Training curves use brax's *stochastic* eval, so the
teacher and PPO-linear are also evaluated stochastically for reference.

Reading the result (per n):
    distilled linear ≫ PPO-linear  -> (b): PPO under-optimises the linear class
    distilled linear ≈ PPO-linear  -> (a): the linear class is the limit
    MLP control ≪ teacher          -> pipeline lossy, result inconclusive

Outputs (--out-dir): results.json (all numbers), distill_summary.png, and a
printed table.

Usage
-----
    experiment/mujoco_playground/.venv/bin/python experiment/original_idea/distill_linear.py
    # smoke test:
    ... distill_linear.py --n-list 1 --seeds 0 --dagger-iters 1 --no-mlp-control
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import optax
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "rl"))
import directed_locomotion_ppo_jax as dl  # noqa: E402  (registers the env)


# ------------------------------------------------------------------------- #
#                         Env + PPO policy loading                          #
# ------------------------------------------------------------------------- #

def make_env(run_dir: Path, n_worlds: int):
    cfg = config_dict.ConfigDict(json.loads((run_dir / "env_config.json").read_text()))
    cfg.naconmax = n_worlds * dl.NACONMAX_PER_ENV
    cfg.naccdmax = n_worlds * dl.NACCDMAX_PER_ENV
    return dl.DirectedLocomotionInsect(cfg)


def load_ppo_policy(run_dir: Path, env, deterministic: bool):
    nf = json.loads((run_dir / "ppo_config.json").read_text())["network_factory"]
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(nf["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(nf["value_hidden_layer_sizes"]),
        policy_obs_key=nf["policy_obs_key"],
        value_obs_key=nf["value_obs_key"],
    )
    params = model.load_params(str(run_dir / "params.pkl"))
    pol = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=deterministic))
    return lambda obs, key: pol(obs, key)[0]  # drop brax's extras dict


# ------------------------------------------------------------------------- #
#                               Rollouts                                    #
# ------------------------------------------------------------------------- #

class Harness:
    """Vmapped raw-env rollouts with a fixed number of parallel episodes."""

    def __init__(self, env, n_envs: int, episode_length: int):
        self.env, self.n, self.T = env, n_envs, episode_length
        self.reset = jax.jit(jax.vmap(env.reset))
        self.step = jax.jit(jax.vmap(env.step))

    def run(self, driver, seed: int, labeler=None):
        """driver/labeler: (obs_dict_batch, key) -> actions batch.

        Returns targets reached per episode (n_envs,), and if labeler is given
        the visited 'state' observations, labels and alive mask.
        """
        key = jax.random.PRNGKey(seed)
        key, rk = jax.random.split(key)
        state = self.reset(jax.random.split(rk, self.n))
        alive = np.ones(self.n, bool)
        reached = np.zeros(self.n)
        X, Y, M = [], [], []
        for _ in range(self.T):
            key, k1, k2 = jax.random.split(key, 3)
            act = driver(state.obs, k1)
            if labeler is not None:
                X.append(np.asarray(state.obs["state"]))
                Y.append(np.asarray(labeler(state.obs, k2)))
                M.append(alive.copy())
            state = self.step(state, act)
            reached += np.asarray(state.metrics["reached"]) * alive
            alive &= ~np.asarray(state.done).astype(bool)
            if not alive.any():
                break
        if labeler is None:
            return reached
        X, Y, M = np.concatenate(X), np.concatenate(Y), np.concatenate(M)
        return reached, X[M], Y[M]

    def evaluate(self, driver, seed: int, reps: int) -> np.ndarray:
        return np.concatenate([self.run(driver, seed + 1000 * r) for r in range(reps)])


# ------------------------------------------------------------------------- #
#                               Students                                    #
# ------------------------------------------------------------------------- #

def _standardiser(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return jp.asarray(mu), jp.asarray(sd)


class LinearStudent:
    name = "linear"

    def init(self, X, Y, key):
        mu, sd = _standardiser(X)
        Xs = (X - np.asarray(mu)) / np.asarray(sd)
        # Ridge regression on atanh(teacher action) as a warm start.
        Z = np.arctanh(np.clip(Y, -0.995, 0.995))
        A = np.hstack([Xs, np.ones((len(Xs), 1))])
        sol = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Z)
        return {"W": jp.asarray(sol[:-1]), "b": jp.asarray(sol[-1]), "mu": mu, "sd": sd}

    @staticmethod
    def pre(p, x):
        return ((x - p["mu"]) / p["sd"]) @ p["W"] + p["b"]

    trainable = ("W", "b")
    steps, lr, batch = 3000, 3e-3, None  # full batch


class MLPStudent:
    name = "mlp_w256_d3"

    def init(self, X, Y, key):
        mu, sd = _standardiser(X)
        sizes = [X.shape[1], 256, 256, 256, Y.shape[1]]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            key, k = jax.random.split(key)
            layers.append({"w": jax.random.normal(k, (a, b)) * np.sqrt(1.0 / a),
                           "b": jp.zeros(b)})
        return {"layers": layers, "mu": mu, "sd": sd}

    @staticmethod
    def pre(p, x):
        h = (x - p["mu"]) / p["sd"]
        for layer in p["layers"][:-1]:
            h = jax.nn.silu(h @ layer["w"] + layer["b"])
        return h @ p["layers"][-1]["w"] + p["layers"][-1]["b"]

    trainable = ("layers",)
    steps, lr, batch = 6000, 1e-3, 4096


def fit(student, params, X, Y, key):
    """Adam on ‖tanh(pre(x)) − y‖²; mu/sd stay fixed."""
    train = {k: params[k] for k in student.trainable}
    fixed = {k: v for k, v in params.items() if k not in student.trainable}
    opt = optax.adam(student.lr)
    st = opt.init(train)
    X, Y = jp.asarray(X), jp.asarray(Y)

    def loss(tr, xb, yb):
        return jp.mean((jp.tanh(student.pre({**tr, **fixed}, xb)) - yb) ** 2)

    @jax.jit
    def update(tr, st, key):
        if student.batch is None:
            xb, yb = X, Y
        else:
            idx = jax.random.randint(key, (student.batch,), 0, X.shape[0])
            xb, yb = X[idx], Y[idx]
        l, g = jax.value_and_grad(loss)(tr, xb, yb)
        upd, st = opt.update(g, st)
        return optax.apply_updates(tr, upd), st, l

    for _ in range(student.steps):
        key, k = jax.random.split(key)
        train, st, _ = update(train, st, k)
    final = float(jax.jit(loss)(train, X, Y))
    return {**train, **fixed}, final


def student_driver(student, params):
    f = jax.jit(lambda p, obs: jp.tanh(student.pre(p, obs["state"])))
    return lambda obs, key: f(params, obs)


def dagger(student, harness, teacher, iters, seed, eval_reps, log):
    key = jax.random.PRNGKey(seed)
    reached, X, Y = harness.run(teacher, seed, labeler=teacher)
    key, k = jax.random.split(key)
    params = student.init(X, Y, k)
    curve = []
    for it in range(iters + 1):
        key, k = jax.random.split(key)
        params, mse = fit(student, params, X, Y, k)
        drv = student_driver(student, params)
        score = harness.evaluate(drv, 10_000 + seed, eval_reps)
        curve.append({"iter": it, "n_samples": int(len(X)), "mse": mse,
                      "targets_mean": float(score.mean())})
        log(f"    {student.name} iter {it}: samples={len(X):>7}  mse={mse:.4f}  "
            f"targets/ep={score.mean():.2f}")
        if it == iters:
            break
        # Student drives, teacher labels; aggregate.
        _, Xn, Yn = harness.run(drv, seed + 100 * (it + 1), labeler=teacher)
        X, Y = np.concatenate([X, Xn]), np.concatenate([Y, Yn])
    return curve


# ------------------------------------------------------------------------- #
#                                   Main                                    #
# ------------------------------------------------------------------------- #

def find_run(data_dir: Path, n, w, d, s):
    hits = sorted(data_dir.glob(f"insect_n_n{n}_w{w}_d{d}_seed{s}-*/params.pkl"))
    return hits[-1].parent if hits else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    data = ROOT.parent / "__data__"
    p.add_argument("--teacher-dir", type=Path, default=data / "directed_capacity_sweep")
    p.add_argument("--linear100-dir", type=Path, default=data / "directed_capacity_sweep")
    p.add_argument("--linear300-dir", type=Path,
                   default=data / "directed_capacity_sweep_linear300M")
    p.add_argument("--out-dir", type=Path, default=ROOT / "__distill__")
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--dagger-iters", type=int, default=6)
    p.add_argument("--n-envs", type=int, default=128)
    p.add_argument("--eval-reps", type=int, default=2,
                   help="Eval episodes = n_envs × eval_reps (default 256).")
    p.add_argument("--no-mlp-control", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res_path = args.out_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    for n in args.n_list:
        env = harness = None
        for s in args.seeds:
            tag = f"n{n}_seed{s}"
            teacher_dir = find_run(args.teacher_dir, n, 256, 3, s)
            if teacher_dir is None:
                log(f"{tag}: no w256 d3 teacher, skipping")
                continue
            if tag in results and results[tag].get("complete"):
                log(f"{tag}: done already, skipping")
                continue
            if harness is None:
                env = make_env(teacher_dir, args.n_envs)
                harness = Harness(env, args.n_envs, int(env._config.episode_length))
            log(f"{tag}: teacher {teacher_dir.name}")
            r = {"n": n, "seed": s, "teacher_dir": teacher_dir.name}
            ev_seed = 10_000 + s

            for mode, det in (("det", True), ("stoch", False)):
                pol = load_ppo_policy(teacher_dir, env, det)
                r[f"teacher_{mode}"] = float(harness.evaluate(pol, ev_seed, args.eval_reps).mean())
                for lab, d in (("ppo_linear_100M", args.linear100_dir),
                               ("ppo_linear_300M", args.linear300_dir)):
                    ld = find_run(d, n, 16, 0, s)
                    if ld is not None:
                        lp = load_ppo_policy(ld, env, det)
                        r[f"{lab}_{mode}"] = float(
                            harness.evaluate(lp, ev_seed, args.eval_reps).mean())
            log("  " + "  ".join(f"{k}={v:.2f}" for k, v in r.items()
                                 if isinstance(v, float)))

            teacher = load_ppo_policy(teacher_dir, env, True)
            students = [LinearStudent()] + ([] if args.no_mlp_control else [MLPStudent()])
            for st in students:
                curve = dagger(st, harness, teacher, args.dagger_iters, s, args.eval_reps, log)
                r[f"distill_{st.name}_curve"] = curve
                r[f"distill_{st.name}"] = curve[-1]["targets_mean"]
            r["complete"] = True
            results[tag] = r
            res_path.write_text(json.dumps(results, indent=2))

    summarise(results, args.out_dir)


def summarise(results: dict, out_dir: Path) -> None:
    cols = [("teacher_det", "teacher w256 d3 (PPO)", "tab:blue"),
            ("distill_mlp_w256_d3", "MLP student (control)", "tab:cyan"),
            ("distill_linear", "distilled linear (DAgger)", "tab:orange"),
            ("ppo_linear_300M_det", "PPO linear, 300M", "0.3"),
            ("ppo_linear_100M_det", "PPO linear, 100M", "0.65")]
    ns = sorted({v["n"] for v in results.values() if v.get("complete")})
    print("\nmean over seeds, deterministic eval, targets reached / episode")
    print("n   " + "".join(f"{lab[:26]:>28}" for _, lab, _ in cols))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(cols)
    for q, (key, lab, col) in enumerate(cols):
        means = []
        for i, n in enumerate(ns):
            vals = [v[key] for v in results.values()
                    if v.get("complete") and v["n"] == n and key in v]
            m = float(np.mean(vals)) if vals else np.nan
            means.append(m)
            x = i + (q - (len(cols) - 1) / 2) * width
            if vals:
                ax.bar(x, m, width, color=col, label=lab if i == 0 else None)
                ax.scatter(np.full(len(vals), x), vals, color="k", s=8, zorder=3)
        cols_vals = means
        results.setdefault("_summary", {})[key] = dict(zip(map(str, ns), cols_vals))
    for i, n in enumerate(ns):
        print(f"{n:<4}" + "".join(
            f"{results['_summary'][k][str(n)]:>28.2f}" for k, _, _ in cols))
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("targets reached / episode\n(deterministic eval)")
    ax.set_title("Is PPO's linear policy the best linear policy?\n"
                 "bars = mean over teacher seeds, dots = seeds", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "distill_summary.png", dpi=150)
    plt.close(fig)
    results.pop("_summary", None)
    print(f"\nwrote {out_dir}/results.json and distill_summary.png")


if __name__ == "__main__":
    main()
