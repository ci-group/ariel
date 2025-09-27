"""TODO(jmdm): description of script."""

# Third-party libraries
import mujoco
import numpy as np

# Global constants
SEED = 42

# Global functions
RNG = np.random.default_rng(SEED)


def simple_runner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    steps_per_loop: int = 100,
) -> None:
    """
    Run a simple headless simulation for a given duration.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to simulate.
    data : mujoco.MjData
        The MuJoCo data to simulate.
    duration : float, optional
        The duration of the simulation in seconds, by default 10.0
    steps_per_loop : int, optional
        The number of simulation steps to take in each loop, by default 100
    """
    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Define action specification and set policy
    data.ctrl = RNG.normal(scale=0.1, size=model.nu)  # type: ignore

    while data.time < duration:
        mujoco.mj_step(model, data, nstep=steps_per_loop)
