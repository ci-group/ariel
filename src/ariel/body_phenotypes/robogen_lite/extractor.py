"""TODO(jmdm): description of script."""

import mujoco
import numpy as np


class RoboGenCoreExtractor:
    """Extractor for core geoms in a Mujoco simulation with RoboGenLite robots."""

    def __init__(self, model, data, world):
        """Initialize the extractor with model, data, and world.

        Parameters
        ----------
        model : mujoco.MjModel
            The Mujoco model.
        data : mujoco.MjData
            The Mujoco data.
        world : mujoco.MjvScene
            The Mujoco world scene."""
        self.model = model
        self.data = data
        self.world = world
        self.geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)

    def track_core_geoms(self) -> list:
        """Extract core geoms from mujoco data.

        Returns
        -------
        list
            List of core geoms."""
        return [
            self.data.bind(geom) for geom in self.geoms if "core" in geom.name
        ]

    def get_core_path(self) -> np.ndarray:
        """Get the path of core geoms -> [xpos, xpos, ...]

        Returns
        -------
        np.ndarray
            Numpy Array of core geom positions."""
        to_track = self.track_core_geoms()
        if not to_track:
            return np.array([])

        xy_history = np.array([geom.xpos for geom in to_track])
        return xy_history

    def xyz_displacement_extractor(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the first and last core geom xpos for displacement calculation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of first and last core geom positions."""
        core_path = self.get_core_path()

        return core_path[0], core_path[-1] if core_path.size > 0 else (
            None,
            None,
        )

    def xy_displacement_extractor(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the first and last core geom (x,y)-position for displacement calculation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of first and last core geom (x,y)-positions."""
        core_path = self.get_core_path()

        return core_path[0, :2], core_path[-1, :2] if core_path.size > 0 else (
            None,
            None,
        )
