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
    data.ctrl = RNG.normal(scale=0.1, size=model.nu)

    while data.time < duration:
        mujoco.mj_step(model, data, nstep=steps_per_loop)


def thread_safe_runner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller, 
    duration: float = 10.0,
) -> None:
    """
    Thread-safe runner for custom controllers.
    
    Applies the controller logic explicitly at every step, avoiding 
    the need for the thread-unsafe global `mujoco.set_mjcb_control`.
    """
    # Calculate the total number of steps based on the simulation timestep
    total_steps = int(duration / model.opt.timestep)
    
    for _ in range(total_steps):
        # 1. Update the control inputs manually using your neural network
        controller.set_control(model, data)
        
        # 2. Step the physics simulation forward by exactly one timestep
        mujoco.mj_step(model, data)