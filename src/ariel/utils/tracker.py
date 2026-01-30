"""TODO(jmdm): description of script."""

# Standard library
from typing import Any

# Third-party libraries
import mujoco as mj

# Local libraries
from ariel import log


class Tracker:
    def __init__(
        self,
        mujoco_obj_to_find: mj.mjtObj | None = None,
        name_to_bind: str | None = None,
        observable_attributes: list[str] | None = None,
    ) -> None:
        """
        A utility class to track and log the state of specified MuJoCo objects.
        Automatically updates during simulation steps.

        Parameters
        ----------
        mujoco_obj_to_find
            The type of MuJoCo object to find and track (e.g., mjtObj.mjOBJ_GEOM).
            If None, defaults to tracking all geoms.
        name_to_bind
            A substring to match in the names of the objects to bind and track.
            If None, defaults to "core".
        observable_attributes
            A list of attribute names to track for each bound object (e.g., ["xpos", "xquat"]).
            If None, defaults to tracking the "xpos" attribute.
        """
        # Set default tracking parameters
        if mujoco_obj_to_find is None or name_to_bind is None:
            mujoco_obj_to_find = mj.mjtObj.mjOBJ_GEOM
            name_to_bind = "core"
            msg = "No tracking parameters provided, "
            msg += "defaulting to tracking all geoms with 'core' in their name."
            log.info(msg)

        # Set default observable attributes
        if observable_attributes is None:
            observable_attributes = ["xpos"]

        # Save local variables
        self.mujoco_obj_to_find = mujoco_obj_to_find
        self.name_to_bind = name_to_bind
        self.observable_attributes = observable_attributes
        self.history: dict[str, dict[int, list[Any]]] = {}

    def setup(
        self,
        world: mj.MjSpec,
        data: mj.MjData,
    ) -> None:
        """
        Setup the tracker by finding and binding the specified MuJoCo objects.

        Parameters
        ----------
        world : mj.MjSpec
            The MuJoCo model specification to search for objects.
        data : mj.MjData
            The MuJoCo data to bind the found objects.
        """
        # Find all objects of the specified type and bind them
        self.geoms = world.worldbody.find_all(self.mujoco_obj_to_find)
        self.to_track = [
            data.bind(geom)
            for geom in self.geoms
            if self.name_to_bind in geom.name
        ]

        # Initialize history dictionary
        for attr in self.observable_attributes:
            if attr not in self.history:
                self.history[attr] = {
                    idx: [] for idx in range(len(self.to_track))
                }

    def update(self, data: mj.MjData) -> None:
        """
        Update the history of tracked attributes for each bound object.
        """
        # Update the bound objects
        for idx, obj in enumerate(self.to_track):
            for attr in self.observable_attributes:
                self.history[attr][idx].append(getattr(obj, attr).copy())

    def reset(self) -> None:
        """
        Reset the history of tracked attributes.
        """

        # Reset the history dictionary
        for attr in self.observable_attributes:
            for idx in range(len(self.to_track)):
                self.history[attr][idx] = []
