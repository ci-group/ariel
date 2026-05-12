"""Companion module for mujoco_worker.ipynb.

Defines TargetedLocomotionWorker and LocomotionConfig as module-level
classes so they can be pickled correctly by spawned worker processes on
Windows. Classes defined inside Jupyter notebook cells live in __main__,
which spawned processes cannot import.
"""

from dataclasses import dataclass

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch

from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase


class _Network:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.W1 = torch.randn(hidden_size, input_size)
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(output_size, hidden_size)
        self.b2 = torch.zeros(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.W1 @ x + self.b1)
        return torch.tanh(self.W2 @ x + self.b2)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


def _fill(net: _Network, params: np.ndarray) -> None:
    offset = 0
    for p in net.parameters():
        n = p.numel()
        p.data = torch.tensor(params[offset:offset + n], dtype=torch.float32).reshape(p.shape)
        offset += n


@dataclass
class LocomotionConfig(EvalConfig):
    """EvalConfig extended with CMA-ES and controller parameters."""

    cma_generations: int = 10
    cma_pop_size: int = 5
    hidden_size: int = 16


class TargetedLocomotionWorker(MuJoCoWorkerBase):
    """Evaluates a robot on a targeted-locomotion task.

    Fitness = distance to target after CMA-ES controller optimisation
    (lower is better).
    """

    def evaluate(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        config: LocomotionConfig,
    ) -> float:
        local_rng = np.random.default_rng(config.seed)

        state_size = len(data.qpos) + len(data.qvel) + 3
        net = _Network(state_size, config.hidden_size, model.nu)
        num_vars = sum(p.numel() for p in net.parameters())
        opt = ng.optimizers.CMA(
            parametrization=num_vars,
            budget=config.cma_generations * config.cma_pop_size,
        )
        best = float("inf")

        for _ in range(config.cma_generations):
            candidates = [opt.ask() for _ in range(config.cma_pop_size)]
            for cand in candidates:
                _fill(net, cand.value)
                mj.mj_resetData(model, data)
                data.ctrl[:] = local_rng.normal(scale=0.1, size=model.nu)
                mj.mj_step(model, data, nstep=300)
                displacement = data.qpos[:3].copy()
                target = tuple(np.array(config.target_position) + displacement)

                def cb(m, d, _target=target, _net=net) -> None:
                    pos = d.qpos[:3].copy()
                    vec = np.array(_target) - pos
                    dist = np.linalg.norm(vec) + 1e-6
                    state = np.concatenate([d.qpos.copy(), d.qvel.copy(), vec / dist])
                    ctrl = _net.forward(torch.tensor(state, dtype=torch.float32))
                    d.ctrl[:] = ctrl.detach().numpy()[:m.nu]

                mj.set_mjcb_control(cb)
                mj.mj_step(model, data, nstep=1_000)
                mj.set_mjcb_control(None)

                fitness = float(np.linalg.norm(np.array(target) - data.qpos[:3]))
                opt.tell(cand, fitness)
                best = min(best, fitness)

        return best
