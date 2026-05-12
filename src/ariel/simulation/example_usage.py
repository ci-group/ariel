"""Example: parallel robot evaluation using MuJoCoWorkerBase.

Shows how to subclass MuJoCoWorkerBase for a targeted-locomotion task
with a CMA-ES-optimised ANN controller, then run it through a process pool.
"""

from dataclasses import dataclass
from multiprocessing import get_context

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch

from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase

# ---------------------------------------------------------------------------
# Minimal ANN controller (experiment-specific)
# ---------------------------------------------------------------------------


class _Network:
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 ) -> None:
        self.W1 = torch.randn(hidden_size, input_size)
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(output_size, hidden_size)
        self.b2 = torch.zeros(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.W1 @ x + self.b1)
        return torch.tanh(self.W2 @ x + self.b2)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


def _fill_parameters(net: _Network, params: np.ndarray) -> None:
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(
            params[offset: offset + numel], dtype=torch.float32
        ).reshape(p.shape)
        offset += numel


# ---------------------------------------------------------------------------
# Experiment-specific EvalConfig (optional — plain EvalConfig also works)
# ---------------------------------------------------------------------------

@dataclass
class LocomotionConfig(EvalConfig):
    """Experiment-specific config for the targeted-locomotion task.

    Carries the same fields as EvalConfig, plus additional ones for
    CMA-ES and the ANN controller.
    """

    cma_generations: int = 20
    cma_pop_size: int = 5
    hidden_size: int = 16


# ---------------------------------------------------------------------------
# Worker subclass
# ---------------------------------------------------------------------------

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
        rng = np.random.default_rng(config.seed)

        state_size = len(data.qpos) + len(data.qvel) + 3
        net = _Network(state_size, config.hidden_size, model.nu)
        num_vars = sum(p.numel() for p in net.parameters())

        opt = ng.optimizers.CMA(
            parametrization=num_vars,
            budget=config.cma_generations * config.cma_pop_size,
        )

        min_fitness = float("inf")

        for _ in range(config.cma_generations):
            candidates = [opt.ask() for _ in range(config.cma_pop_size)]
            for cand in candidates:
                _fill_parameters(net, cand.value)
                mj.mj_resetData(model, data)

                # Warmup
                data.ctrl[:] = rng.normal(scale=0.1, size=model.nu)
                mj.mj_step(model, data, nstep=300)

                displacement = data.qpos[:3].copy()
                target = tuple(np.array(config.target_position) + displacement)

                def cb(m, d) -> None:
                    pos = d.qpos[:3].copy()
                    vec = np.array(target) - pos
                    dist = np.linalg.norm(vec) + 1e-6
                    state = np.concatenate([d.qpos.copy(), d.qvel.copy(), vec / dist])
                    ctrl = net.forward(torch.tensor(state, dtype=torch.float32))
                    d.ctrl[:] = ctrl.detach().numpy()[: m.nu]

                mj.set_mjcb_control(cb)
                mj.mj_step(model, data, nstep=1_000)
                mj.set_mjcb_control(None)

                fitness = float(
                    np.linalg.norm(np.array(target) - data.qpos[:3])
                )
                opt.tell(cand, fitness)
                min_fitness = min(min_fitness, fitness)

        return min_fitness


# ---------------------------------------------------------------------------
# How to use it in an evolution loop
# ---------------------------------------------------------------------------

def evaluate_population(
    eval_args: list[tuple[str, EvalConfig]],
    worker: MuJoCoWorkerBase,
    num_workers: int = 8,
) -> list[float]:
    """Run eval_args through a process pool and return fitness values."""
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        return list(pool.imap(worker, eval_args, chunksize=1))


if __name__ == "__main__":
    # Dummy smoke-test: two identical placeholder XML strings.
    # Replace with real robot XML from construct_mjspec_from_graph + SimpleFlatWorld.
    dummy_xml = """
    <mujoco>
      <worldbody>
        <body name="core" pos="0 0 0.1">
          <joint type="free"/>
          <geom type="box" size="0.1 0.1 0.05" mass="1"/>
          <body name="leg" pos="0.1 0 0">
            <joint name="hinge" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.1"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="hinge" gear="10"/>
      </actuator>
    </mujoco>
    """

    config = LocomotionConfig(
        spawn_position=(0.0, 0.0, 0.1),
        target_position=(2.0, 0.0, 0.1),
        seed=42,
    )

    eval_args = [(dummy_xml, config), (dummy_xml, config)]
    worker = TargetedLocomotionWorker()

    results = evaluate_population(eval_args, worker, num_workers=2)
    print("Fitness values:", results)
