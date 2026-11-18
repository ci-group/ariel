"""GNN controller for insect_n morphologies with PPO training.

Extends undirected_locomotion_ppo_jax.py by replacing the flat MLP policy with
a Graph Neural Network that mirrors the robot's kinematic tree.  Because every
GNN sub-MLP (message, update, output) is applied independently per-node or
per-edge, all parameter tensors have shapes that are independent of the number
of nodes/edges.  The *same* parameter set therefore applies to any
insect_n(n_segments) size, enabling zero-shot transfer or warm-start
fine-tuning from a checkpoint trained on a smaller morphology.

Graph structure for insect_n(n_segments)
-----------------------------------------
Node layout  (n_nodes = nu + 1 = 3*n_segments + 3)::

    0              : torso (no actuator, carries global state)
    1..n_seg       : spine hinges H0..H_{n_seg-1}   → actuators 0..n_seg-1
    n_seg+1        : core right leg                  → actuator  n_seg
    n_seg+2        : core left leg                   → actuator  n_seg+1
    n_seg+3+2k     : spine-brick k right leg         → actuator  n_seg+2+2k
    n_seg+4+2k     : spine-brick k left leg          → actuator  n_seg+3+2k
                     (k = 0 .. n_seg-1)

Edges (bidirectional)::

    torso  <->  spine_hinge_0
    spine_hinge_k  <->  spine_hinge_{k+1}
    torso  <->  core_right  <->  core_left
    spine_hinge_k  <->  SBk_right  <->  SBk_left

Node features (NODE_FEAT_DIM = 24, morphology-independent)::

    [:3]  local: [joint_angle, joint_vel, last_act]  (zeros for torso)
    [3:]  global: [height(1), rot(9), linvel(3), angvel(3), up(3), phase(2)]

Transfer learning
-----------------
Since no GNN Dense layer depends on n_segments, loading params trained on
insect_n(small) into the network built for insect_n(large) is a direct
assignment — no slicing or zero-padding needed.  Only the obs-normalizer
(which tracks per-dimension statistics for the flat 21+3*nu obs vector)
must be re-initialised.  Pass ``--transfer-from /path/to/params.pkl`` to
enable the warm start automatically.

Usage
-----
Train GNN on insect_n(2)::

    uv run experiment/rl/undirected_locomotion_ppo_gnn_jax.py \\
        --morphology insect_n --n-pairs 2 --num-timesteps 20_000_000

Warm-start on insect_n(4) from an insect_n(2) checkpoint::

    uv run experiment/rl/undirected_locomotion_ppo_gnn_jax.py \\
        --morphology insect_n --n-pairs 4 \\
        --transfer-from /path/to/insect_n_n2_seed0/params.pkl
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from brax.io import model as brax_model
from brax.training import distribution, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_net_lib
from brax.training.agents.ppo import train as ppo
from brax.training.networks import FeedForwardNetwork, normalizer_select
from flax import linen as nn
from ml_collections import config_dict

# ---- pull everything env-related from the existing module -----------------
sys.path.insert(0, str(Path(__file__).parent))
from undirected_locomotion_ppo_jax import (  # noqa: E402
    UndirectedLocomotionInsect,
    default_config,
    ppo_config,
    NUM_ENVS,
    NUM_MINIBATCHES,
    BATCH_RATIO,
    BATCH_SIZE,
    NJMAX_PER_ENV,
    NACONMAX_PER_ENV,
    NACCDMAX_PER_ENV,
    render_rollout,
)

from mujoco_playground import wrapper  # noqa: E402


# ============================================================================ #
#                           Kinematic graph                                    #
# ============================================================================ #

NODE_FEAT_DIM = 24  # 3 local + 21 global — fixed regardless of morphology
GLOBAL_FEAT_DIM = 21  # height(1) + rot(9) + linvel(3) + angvel(3) + up(3) + phase(2)
LOCAL_FEAT_DIM = 3   # joint_angle, joint_vel, last_act


def build_insect_graph(n_segments: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (senders, receivers, n_nodes) for insect_n(n_segments).

    Node IDs are assigned so that node ``i+1`` corresponds to actuator ``i``
    (i.e. slicing ``h[:, 1:, :]`` after message passing gives per-actuator
    hidden states in the same order as MuJoCo's actuator array).
    """
    nu = 3 * n_segments + 2
    n_nodes = nu + 1

    edges: list[tuple[int, int]] = []

    def add_undirected(i: int, j: int) -> None:
        edges.append((i, j))
        edges.append((j, i))

    # Torso → spine chain
    if n_segments > 0:
        add_undirected(0, 1)
    for k in range(n_segments - 1):
        add_undirected(k + 1, k + 2)

    # Torso → core legs + sibling
    core_r = n_segments + 1
    core_l = n_segments + 2
    add_undirected(0, core_r)
    add_undirected(0, core_l)
    add_undirected(core_r, core_l)

    # Each spine hinge → its spine-brick's leg pair + sibling
    for k in range(n_segments):
        sb_r = n_segments + 3 + 2 * k
        sb_l = n_segments + 4 + 2 * k
        add_undirected(k + 1, sb_r)
        add_undirected(k + 1, sb_l)
        add_undirected(sb_r, sb_l)

    senders = np.array([e[0] for e in edges], dtype=np.int32)
    receivers = np.array([e[1] for e in edges], dtype=np.int32)
    return senders, receivers, n_nodes


# ============================================================================ #
#                           Flax GNN modules                                  #
# ============================================================================ #

def _obs_to_node_features(obs: jnp.ndarray, nu: int) -> jnp.ndarray:
    """Split flat obs into per-node feature matrix.

    Args:
        obs:  (batch, 21+3*nu)  normalised flat observation.
        nu:   total number of actuators.

    Returns:
        node_feats:  (batch, nu+1, NODE_FEAT_DIM)
            Row 0 is the torso node (zeros for local part).
            Row i (i>=1) is actuator i-1.
    """
    batch = obs.shape[0]

    height     = obs[:, 0:1]
    rot        = obs[:, 1:10]
    j_ang      = obs[:, 10 : 10 + nu]
    j_vel      = obs[:, 10 + nu : 10 + 2 * nu]
    linvel     = obs[:, 10 + 2 * nu : 10 + 2 * nu + 3]
    angvel     = obs[:, 10 + 2 * nu + 3 : 10 + 2 * nu + 6]
    up         = obs[:, 10 + 2 * nu + 6 : 10 + 2 * nu + 9]
    last_act   = obs[:, 10 + 2 * nu + 9 : 10 + 3 * nu + 9]
    phase      = obs[:, 10 + 3 * nu + 9 :]  # 2 dims

    global_feat = jnp.concatenate(
        [height, rot, linvel, angvel, up, phase], axis=-1
    )  # (batch, 21)

    # Actuator nodes: (batch, nu, 3) local ++ (batch, nu, 21) global
    local_feats = jnp.stack([j_ang, j_vel, last_act], axis=-1)  # (batch, nu, 3)
    global_tiled = jnp.tile(global_feat[:, None, :], (1, nu, 1))   # (batch, nu, 21)
    act_nodes = jnp.concatenate([local_feats, global_tiled], axis=-1)  # (batch, nu, 24)

    # Torso node: zero local part
    torso_local = jnp.zeros((batch, 1, LOCAL_FEAT_DIM))
    torso_node = jnp.concatenate(
        [torso_local, global_feat[:, None, :]], axis=-1
    )  # (batch, 1, 24)

    return jnp.concatenate([torso_node, act_nodes], axis=1)  # (batch, nu+1, 24)


class _GNNEncoder(nn.Module):
    """Shared GNN trunk: embed → K × message-pass → (batch, n_nodes, hidden_dim).

    Parameters are morphology-independent: the same Dense weights apply to any
    graph with the same NODE_FEAT_DIM.  Graph topology (senders/receivers) is
    captured from the outer closure and treated as a compile-time constant.
    """

    hidden_dim: int
    n_mp_steps: int
    nu: int
    senders: tuple  # tuple[int, ...] stored as Python tuple for hashability
    receivers: tuple
    n_nodes: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (batch, obs_dim)
        node_feats = _obs_to_node_features(obs, self.nu)  # (batch, n_nodes, 24)

        s = jnp.array(self.senders)    # (n_edges,) — compile-time constant
        r = jnp.array(self.receivers)  # (n_edges,)
        # Scatter matrix for the batched segment-sum:
        #   one_hot[e, n] = 1 iff edge e targets node n
        one_hot = jax.nn.one_hot(r, self.n_nodes)  # (n_edges, n_nodes)

        # Initial node embedding
        h = nn.Dense(self.hidden_dim, name="embed")(node_feats)
        h = nn.swish(h)  # (batch, n_nodes, hidden_dim)

        for step in range(self.n_mp_steps):
            # Gather sender / receiver hidden states for every edge
            h_s = h[:, s, :]  # (batch, n_edges, hidden_dim)
            h_r = h[:, r, :]  # (batch, n_edges, hidden_dim)

            # Message: function of both sender and receiver states
            msg = nn.Dense(self.hidden_dim, name=f"msg_{step}")(
                jnp.concatenate([h_s, h_r], axis=-1)
            )
            msg = nn.swish(msg)  # (batch, n_edges, hidden_dim)

            # Scatter-sum: agg[b, n, :] = Σ_{e: r[e]==n} msg[b, e, :]
            # Equivalent to a batched segment_sum without dynamic shapes.
            agg = jnp.einsum("bef,en->bnf", msg, one_hot)  # (batch, n_nodes, hidden_dim)

            # Node update with residual connection
            h_new = nn.Dense(self.hidden_dim, name=f"update_{step}")(
                jnp.concatenate([h, agg], axis=-1)
            )
            h = h + nn.swish(h_new)  # (batch, n_nodes, hidden_dim)

        return h


class GNNPolicyModule(nn.Module):
    """GNN policy head.  Output: (batch, 2*nu) = [mean_0..mean_{nu-1}, scale_0..scale_{nu-1}].

    Compatible with brax's NormalTanhDistribution which splits params at the
    midpoint into (loc, raw_scale).
    """

    hidden_dim: int
    n_mp_steps: int
    nu: int
    senders: tuple
    receivers: tuple
    n_nodes: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        squeeze = obs.ndim == 1
        obs = jnp.atleast_2d(obs)  # guarantee (batch, obs_dim)

        h = _GNNEncoder(
            hidden_dim=self.hidden_dim,
            n_mp_steps=self.n_mp_steps,
            nu=self.nu,
            senders=self.senders,
            receivers=self.receivers,
            n_nodes=self.n_nodes,
            name="encoder",
        )(obs)  # (batch, n_nodes, hidden_dim)

        # Nodes 1..n_nodes-1 are actuators (in MuJoCo actuator order)
        act_h = h[:, 1:, :]  # (batch, nu, hidden_dim)

        # Per-actuator output: [mean_i, raw_scale_i]
        out = nn.Dense(2, name="policy_out")(act_h)  # (batch, nu, 2)

        # Reshape to brax convention: first half = loc, second half = scale
        mean  = out[:, :, 0]   # (batch, nu)
        scale = out[:, :, 1]   # (batch, nu)
        result = jnp.concatenate([mean, scale], axis=-1)  # (batch, 2*nu)

        return result[0] if squeeze else result


class GNNValueModule(nn.Module):
    """GNN value head.  Output: (batch,) scalar baseline estimate.

    Uses global mean-pooling over all node hidden states followed by a small
    MLP, making the value estimate truly size-invariant.
    """

    hidden_dim: int
    n_mp_steps: int
    nu: int
    senders: tuple
    receivers: tuple
    n_nodes: int
    value_mlp_sizes: tuple  # e.g. (64, 64)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        squeeze = obs.ndim == 1
        obs = jnp.atleast_2d(obs)

        h = _GNNEncoder(
            hidden_dim=self.hidden_dim,
            n_mp_steps=self.n_mp_steps,
            nu=self.nu,
            senders=self.senders,
            receivers=self.receivers,
            n_nodes=self.n_nodes,
            name="encoder",
        )(obs)  # (batch, n_nodes, hidden_dim)

        # Global mean pooling over nodes → aggregate morphology-level context
        pooled = jnp.mean(h, axis=1)  # (batch, hidden_dim)

        # Small MLP → scalar value
        x = pooled
        for i, size in enumerate(self.value_mlp_sizes):
            x = nn.swish(nn.Dense(size, name=f"value_mlp_{i}")(x))
        value = nn.Dense(1, name="value_out")(x)  # (batch, 1)
        result = jnp.squeeze(value, axis=-1)       # (batch,)

        return result[0] if squeeze else result


# ============================================================================ #
#                         GNN network factory                                  #
# ============================================================================ #

def make_gnn_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    n_segments: int = 2,
    hidden_dim: int = 64,
    n_mp_steps: int = 3,
    value_mlp_sizes: Sequence[int] = (64, 64),
    obs_key: str = "state",
) -> ppo_net_lib.PPONetworks:
    """Build a PPONetworks object backed by GNN policy and value networks.

    Drop-in replacement for ``ppo_networks.make_ppo_networks``.  Pass as the
    ``network_factory`` argument to ``ppo.train`` (via ``functools.partial``
    with ``n_segments`` pre-filled).

    Args:
        observation_size: as provided by brax/mujoco_playground.
        action_size:      number of actuators (= nu).
        preprocess_observations_fn: brax running-statistics normaliser.
        n_segments:       insect_n spine segment count — determines graph.
        hidden_dim:       GNN node hidden dimension.
        n_mp_steps:       number of message-passing rounds.
        value_mlp_sizes:  hidden sizes for the value-head MLP after pooling.
        obs_key:          key to extract from the observation dict.
    """
    senders_np, receivers_np, n_nodes = build_insect_graph(n_segments)
    senders_t  = tuple(senders_np.tolist())
    receivers_t = tuple(receivers_np.tolist())
    nu = action_size

    # Resolve flat obs dimension from brax's observation_size type
    obs_size_raw = observation_size[obs_key] if isinstance(observation_size, Mapping) else observation_size
    obs_dim: int = jax.tree_util.tree_flatten(obs_size_raw)[0][-1]

    dummy_obs = jnp.zeros((1, obs_dim))

    policy_module = GNNPolicyModule(
        hidden_dim=hidden_dim,
        n_mp_steps=n_mp_steps,
        nu=nu,
        senders=senders_t,
        receivers=receivers_t,
        n_nodes=n_nodes,
    )

    value_module = GNNValueModule(
        hidden_dim=hidden_dim,
        n_mp_steps=n_mp_steps,
        nu=nu,
        senders=senders_t,
        receivers=receivers_t,
        n_nodes=n_nodes,
        value_mlp_sizes=tuple(value_mlp_sizes),
    )

    def policy_apply(processor_params, policy_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[obs_key], normalizer_select(processor_params, obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    def value_apply(processor_params, value_params, obs):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[obs_key], normalizer_select(processor_params, obs_key)
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        return value_module.apply(value_params, obs)

    policy_network = FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=policy_apply,
    )
    value_network = FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs),
        apply=value_apply,
    )

    return ppo_net_lib.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=distribution.NormalTanhDistribution(
            event_size=action_size
        ),
    )


# ============================================================================ #
#                         Transfer-learning utilities                          #
# ============================================================================ #

def create_transfer_params(
    source_path: str,
    target_nu: int,
    std_eps: float = 0.0,
) -> tuple:
    """Build restore_params for warm-starting from a differently-sized checkpoint.

    The GNN policy/value weights are loaded verbatim (all Dense layers have
    morphology-independent shapes).  The obs normalizer is re-initialised from
    scratch for the target morphology's obs_dim = 21 + 3 * target_nu.

    Args:
        source_path:  path to ``params.pkl`` saved by a previous GNN training run.
        target_nu:    actuator count of the new (larger) morphology.
        std_eps:      passed through to running_statistics.init_state.

    Returns:
        (fresh_normalizer, source_policy_params, source_value_params)
        — suitable as ``restore_params`` for ``ppo.train``.
    """
    source_params = brax_model.load_params(source_path)
    # source_params = (normalizer, policy_params, value_params)

    target_obs_dim = 21 + 3 * target_nu
    fresh_normalizer = running_statistics.init_state(
        {"state": specs.Array((target_obs_dim,), np.dtype("float32"))},
        std_eps=std_eps,
    )
    return (fresh_normalizer, source_params[1], source_params[2])


def print_gnn_param_info(params: Any) -> None:
    """Print GNN parameter shapes (useful to verify transfer compatibility)."""
    policy_params = params[1]
    flat = jax.tree_util.tree_leaves_with_path(policy_params)
    print("GNN policy param shapes (all morphology-independent):")
    for path, leaf in flat:
        name = "/".join(str(p.key) for p in path)
        print(f"  {name:60s} {leaf.shape}")


# ============================================================================ #
#                                  Training                                    #
# ============================================================================ #

def train(args: argparse.Namespace) -> tuple[Any, Any, Path]:
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    num_envs = args.num_envs if args.num_envs is not None else NUM_ENVS
    nacc = args.naccdmax_per_env if args.naccdmax_per_env is not None else NACCDMAX_PER_ENV
    nacn = args.naconmax_per_env if args.naconmax_per_env is not None else NACONMAX_PER_ENV
    print(f"NUM_ENVS={num_envs}  NACONMAX/env={nacn}  NACCDMAX/env={nacc}")
    print(
        f"GNN: hidden_dim={args.gnn_hidden_dim}  "
        f"mp_steps={args.gnn_mp_steps}  "
        f"n_segments={args.n_pairs}"
    )

    rl_cfg = ppo_config(args.num_timesteps)
    rl_cfg.num_envs = num_envs
    rl_cfg.batch_size = BATCH_RATIO * num_envs // NUM_MINIBATCHES

    env_cfg = default_config()
    env_cfg.morphology = args.morphology
    env_cfg.n_pairs = args.n_pairs
    env_cfg.naconmax = num_envs * nacn
    env_cfg.naccdmax = num_envs * nacc
    env = UndirectedLocomotionInsect(env_cfg)

    eval_cfg = default_config()
    eval_cfg.morphology = args.morphology
    eval_cfg.n_pairs = args.n_pairs
    eval_cfg.naconmax = int(rl_cfg.num_eval_envs) * nacn
    eval_cfg.naccdmax = int(rl_cfg.num_eval_envs) * nacc
    eval_env = UndirectedLocomotionInsect(eval_cfg)

    nu = env.action_size
    n_segments = args.n_pairs  # insect_n uses n_pairs as n_segments

    tag_bits = [args.morphology]
    if args.morphology in ("centipede", "insect_n", "insect_longspine_n"):
        tag_bits.append(f"n{args.n_pairs}")
    tag_bits += [
        "gnn",
        f"h{args.gnn_hidden_dim}",
        f"k{args.gnn_mp_steps}",
        f"seed{args.seed}",
    ]
    run_tag = "_".join(tag_bits)
    out_dir = (
        Path(args.outdir)
        / f"{run_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    (out_dir / "ppo_config.json").write_text(json.dumps(rl_cfg.to_dict(), indent=2))
    gnn_meta = {
        "hidden_dim": args.gnn_hidden_dim,
        "n_mp_steps": args.gnn_mp_steps,
        "n_segments": n_segments,
        "nu": nu,
        "transfer_from": args.transfer_from,
    }
    (out_dir / "gnn_config.json").write_text(json.dumps(gnn_meta, indent=2))
    print(f"logging to {out_dir}")

    history: dict[str, list[float]] = {"steps": [], "reward": [], "reward_std": []}
    times = [time.monotonic()]
    snapshots: list[tuple[int, Any]] = []

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(time.monotonic())
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        reward_std = float(metrics.get("eval/episode_reward_std", float("nan")))
        history["steps"].append(int(num_steps))
        history["reward"].append(reward)
        history["reward_std"].append(reward_std)
        print(
            f"[{num_steps:>10}] reward={reward:+.3f} ± {reward_std:.3f}"
            f"  elapsed={times[-1] - times[0]:7.1f}s",
            flush=True,
        )

    def snapshot_fn(current_step: int, _make_policy: Any, params_tuple: Any) -> None:
        snapshots.append(
            (int(current_step), jax.tree_util.tree_map(np.asarray, params_tuple))
        )

    train_params = dict(rl_cfg)
    _ = train_params.pop("network_factory")  # discard MLP-specific config
    network_factory = functools.partial(
        make_gnn_ppo_networks,
        n_segments=n_segments,
        hidden_dim=args.gnn_hidden_dim,
        n_mp_steps=args.gnn_mp_steps,
        # value_mlp_sizes left at default (64, 64)
    )
    num_eval_envs = train_params.pop("num_eval_envs")

    # Transfer learning: load GNN body from a smaller morphology
    restore_params = None
    if args.transfer_from is not None:
        print(f"Loading transfer params from: {args.transfer_from}")
        restore_params = create_transfer_params(
            args.transfer_from, target_nu=nu
        )
        print(
            "  → obs normalizer re-initialised for obs_dim=",
            21 + 3 * nu,
            "; GNN body params loaded verbatim",
        )

    train_fn = functools.partial(
        ppo.train,
        **train_params,
        network_factory=network_factory,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        policy_params_fn=snapshot_fn,
        restore_params=restore_params,
        restore_value_fn=True,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
    )

    if len(times) > 1:
        print(f"time to jit: {times[1] - times[0]:.1f}s")
        print(f"time to train: {times[-1] - times[1]:.1f}s")

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    brax_model.save_params(str(out_dir / "params.pkl"), params)
    brax_model.save_params(str(out_dir / "params_snapshots.pkl"), snapshots)

    # Reward curve
    if history["steps"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.array(history["steps"])
        reward = np.array(history["reward"])
        reward_std = np.array(history["reward_std"])
        ax.plot(steps, reward, color="tab:blue", label=f"GNN h={args.gnn_hidden_dim} k={args.gnn_mp_steps}")
        ax.fill_between(steps, reward - reward_std, reward + reward_std, color="tab:blue", alpha=0.25)
        if args.transfer_from:
            ax.set_title(f"insect_n n={n_segments} — GNN (warm-start from n={_infer_source_n(args.transfer_from)})")
        else:
            ax.set_title(f"insect_n n={n_segments} — GNN undirected locomotion (PPO)")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval episode reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "reward_curve.png", dpi=140)
        plt.close(fig)

    return make_inference_fn, params, out_dir


def _infer_source_n(path: str) -> str:
    """Best-effort extraction of n_pairs from a checkpoint directory path."""
    import re
    m = re.search(r"n(\d+)", Path(path).parent.name)
    return m.group(1) if m else "?"


# ============================================================================ #
#                                   Main                                       #
# ============================================================================ #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        default=str(
            Path(__file__).parent.parent / "__data__" / "undirected_locomotion_ppo_gnn_jax"
        ),
    )
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument(
        "--morphology",
        choices=["insect", "centipede", "insect_n", "insect_longspine_n"],
        default="insect_n",
        help="insect_n is the only morphology with a defined GNN graph.",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=2,
        help="Spine segments for insect_n (n_segments).  Also controls GNN size.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--naccdmax-per-env", type=int, default=None)
    parser.add_argument("--naconmax-per-env", type=int, default=None)
    # GNN-specific
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for all GNN Dense layers.",
    )
    parser.add_argument(
        "--gnn-mp-steps",
        type=int,
        default=3,
        help="Number of message-passing rounds.",
    )
    parser.add_argument(
        "--transfer-from",
        default=None,
        metavar="PARAMS_PKL",
        help="Path to params.pkl from a GNN run on a *smaller* morphology.  "
        "GNN body weights are loaded verbatim; obs normalizer is re-initialised.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.morphology not in ("insect_n",):
        print(
            f"WARNING: GNN graph is only defined for insect_n.  "
            f"Got morphology={args.morphology!r}.  "
            "The graph structure may be incorrect for other morphologies."
        )
    make_inference_fn, params, out_dir = train(args)
    import gc
    jax.clear_caches()
    gc.collect()
    try:
        render_rollout(
            make_inference_fn,
            params,
            out_dir,
            args.episode_length,
            args.seed,
            morphology=args.morphology,
            n_pairs=args.n_pairs,
        )
    except Exception as e:
        print(f"render_rollout skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
