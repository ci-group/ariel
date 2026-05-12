"""Base class for parallel MuJoCo robot evaluation.

Users subclass MuJoCoWorkerBase and implement evaluate() with their
experiment-specific fitness logic. The instance is passed directly to
multiprocessing.Pool.imap — it is picklable because it is a named class.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import mujoco as mj
import torch


@dataclass
class EvalConfig:
    """Minimal config passed to each worker process.

    Users may subclass this to carry additional experiment-specific fields.
    """

    spawn_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    seed: int | None = None


class MuJoCoWorkerBase(ABC):
    """Callable base class for MuJoCo robot evaluation in worker processes.

    Subclass and implement evaluate(). Pass an instance to
    multiprocessing.Pool.imap as the map function::

        worker = MyWorker()
        with pool:
            results = list(pool.imap(worker, eval_args))

    where eval_args is a list of (xml_string, EvalConfig) tuples.
    """

    def __call__(self, args: tuple[str, EvalConfig]) -> float:
        """Entry point called by the worker process.

        Loads the MuJoCo model from XML and delegates to evaluate().
        Returns the result of evaluate(), or raises if the XML is invalid.

        Returns
        -------
        float
             The fitness value returned by evaluate().
        """
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        xml_string, config = args
        model = mj.MjModel.from_xml_string(xml_string)
        data = mj.MjData(model)
        return self.evaluate(model, data, config)

    @abstractmethod
    def evaluate(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        config: EvalConfig,
    ) -> float:
        """Evaluate a robot and return a scalar fitness value.

        Called once per individual per generation inside a worker process.
        PyTorch is already limited to 1 thread when this is called.

        Parameters
        ----------
            model: Loaded MuJoCo model.
            data:  Corresponding MjData, ready for simulation.
            config: Experiment configuration passed from the main process.

        Returns
        -------
            A scalar fitness value (convention: lower = better, but the
            sign is entirely up to the user's experiment).
        """
