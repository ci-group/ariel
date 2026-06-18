"""CMA-ES learner wrapping nevergrad for black-box fitness maximisation.

Domain-agnostic — no simulator imports.  Works with any controller that
exposes ``set_theta`` / ``get_theta`` (e.g. ``DistributedMLP``,
``StandardMLP``, ``NaCPG``).
"""

from __future__ import annotations

import numpy as np
import nevergrad as ng


class CMAESLearner:
    """Wraps nevergrad CMA to maximise fitness (nevergrad minimises internally).

    Parameters
    ----------
    n_params:
        Dimensionality of the parameter vector θ.
    init_mean:
        Initial mean for CMA-ES.  Defaults to zeros.
    sigma:
        Initial step-size.
    pop_size:
        Population size (lambda).
    """

    def __init__(
        self,
        n_params: int,
        init_mean: np.ndarray | None = None,
        sigma: float = 0.5,
        pop_size: int = 16,
    ) -> None:
        self._pop_size = pop_size
        self._n_params = n_params
        self._best_theta: np.ndarray = np.zeros(n_params)
        self._best_fitness: float = -float("inf")
        self._all_samples: list[tuple[np.ndarray, float]] = []

        init = (
            np.asarray(init_mean, dtype=np.float64)
            if init_mean is not None
            else np.zeros(n_params, dtype=np.float64)
        )

        param = ng.p.Array(init=init)
        param.set_mutation(sigma=sigma)
        cma_cfg = ng.optimizers.ParametrizedCMA(popsize=pop_size)
        self._opt = cma_cfg(
            parametrization=param,
            budget=int(1e9),
            num_workers=pop_size,
        )
        self._pending: list[ng.p.Parameter] = []

    def ask(self) -> list[np.ndarray]:
        """Return ``pop_size`` candidate θ vectors."""
        self._pending = [self._opt.ask() for _ in range(self._pop_size)]
        return [np.asarray(c.value, dtype=np.float64) for c in self._pending]

    def tell(self, candidates: list[np.ndarray], fitnesses: list[float]) -> None:
        """Record results; fitnesses are maximised."""
        for cand_ng, theta, fitness in zip(self._pending, candidates, fitnesses):
            self._opt.tell(cand_ng, -float(fitness))
            self._all_samples.append((np.asarray(theta, dtype=np.float64), float(fitness)))
            if fitness > self._best_fitness:
                self._best_fitness = fitness
                self._best_theta = np.asarray(theta, dtype=np.float64).copy()
        self._pending = []

    @property
    def best_theta(self) -> np.ndarray:
        return self._best_theta

    @property
    def best_fitness(self) -> float:
        return self._best_fitness

    @property
    def all_samples(self) -> list[tuple[np.ndarray, float]]:
        return self._all_samples
