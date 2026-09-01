"""The mating movement phase of the spatial EA.

Between fitness evaluation and pairing, the whole population is simulated
together in one world. Each robot is driven by its own HyperNEAT substrate and
receives a normalised direction vector as *neural input*. Nothing multiplies
the motor outputs afterwards, so a controller only benefits from the
directional signal if evolution teaches it to use it.

Notes
-----
    * Control is written inside the stepping loop rather than through
      ``mujoco.set_mjcb_control``. The global callback outlives the model it
      closes over, which is a leak across generations.

"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.spatial_ea.hyperneat import (
    CPPN,
    SubstrateNetwork,
    create_substrate_for_gecko,
)
from ariel.spatial_ea.interaction import (
    apply_periodic_boundaries_to_simulation,
    calculate_periodic_displacement,
)

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.individual import SpatialIndividual
    from ariel.spatial_ea.recording import GenerationRecorder
    from ariel.spatial_ea.world import SpawnedPopulation

# Global constants
FREE_JOINT_DOF = 7
SUBSTRATE_WEIGHT_THRESHOLD = 0.2
MIN_DIRECTION_NORM = 0.01
MIN_NEIGHBOUR_DISTANCE = 0.1
NEAREST_NEIGHBOUR_DEAD_ZONE = 0.5
VALID_MOVEMENT_BIASES = (
    "nearest_neighbor",
    "nearest_zone",
    "assigned_zone",
    "none",
)


def build_substrates(
    population: list[SpatialIndividual],
    num_joints: int,
    weight_threshold: float = SUBSTRATE_WEIGHT_THRESHOLD,
) -> list[SubstrateNetwork]:
    """Decode every individual's CPPN into a substrate controller.

    Parameters
    ----------
    population
        Individuals whose genomes should be decoded.
    num_joints
        Number of actuated joints per robot.
    weight_threshold
        Minimum absolute CPPN weight for a substrate connection to exist.

    Returns
    -------
        One substrate network per individual, in population order.
    """
    input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
        num_joints=num_joints,
        use_hidden_layer=True,
        hidden_layer_size=num_joints,
    )

    return [
        SubstrateNetwork(
            input_coords=input_coords,
            hidden_coords=hidden_coords,
            output_coords=output_coords,
            cppn=CPPN(individual.genotype),
            weight_threshold=weight_threshold,
        )
        for individual in population
    ]


def sensor_inputs(
    data: mujoco.MjData,
    robot_idx: int,
    num_joints: int,
    directional_inputs: FloatArray,
) -> FloatArray:
    """Assemble the substrate input vector for one robot.

    The vector is the robot's joint angles, four CPG oscillators at one and two
    hertz in quadrature, the directional inputs, and a constant bias.

    Parameters
    ----------
    data
        Simulation state to read joint angles and time from.
    robot_idx
        Index of the robot within the shared world.
    num_joints
        Number of actuated joints per robot.
    directional_inputs
        Two-element normalised heading toward this robot's target.

    Returns
    -------
        The concatenated input vector.
    """
    offset = robot_idx * (FREE_JOINT_DOF + num_joints) + FREE_JOINT_DOF
    joint_angles = data.qpos[offset : offset + num_joints].copy()

    cpg_inputs = np.array([
        np.sin(data.time * 1.0),
        np.cos(data.time * 1.0),
        np.sin(data.time * 2.0),
        np.cos(data.time * 2.0),
    ])

    return np.concatenate([
        joint_angles,
        cpg_inputs,
        directional_inputs,
        np.array([1.0]),
    ])


@dataclass
class MatingController:
    """Drives a whole population toward its mating targets.

    Parameters
    ----------
    population
        Individuals being controlled, in spawn order.
    substrates
        One decoded substrate network per individual.
    num_joints
        Number of actuated joints per robot.
    control_clip_min
        Lower bound applied to every actuator command.
    control_clip_max
        Upper bound applied to every actuator command.
    movement_bias
        Source of the directional input. One of
        :data:`VALID_MOVEMENT_BIASES`.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether headings take the shortest path around the world edges.
    mating_zone_centers
        Zone centres, required by the zone-based biases.
    assigned_zones
        Mapping from ``unique_id`` to zone index, used by the
        ``assigned_zone`` bias.
    """

    population: list[SpatialIndividual]
    substrates: list[SubstrateNetwork]
    num_joints: int
    control_clip_min: float
    control_clip_max: float
    movement_bias: str = "nearest_neighbor"
    world_size: tuple[float, float] = (10.0, 10.0)
    use_periodic_boundaries: bool = False
    mating_zone_centers: list[tuple[float, float]] = field(
        default_factory=list,
    )
    assigned_zones: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the requested bias has the data it needs.

        Raises
        ------
        ValueError
            If the bias is unknown, or if a zone-based bias was requested
            without any mating zones.
        """
        if self.movement_bias not in VALID_MOVEMENT_BIASES:
            msg = (
                f"Unknown movement_bias {self.movement_bias!r}; "
                f"expected one of {VALID_MOVEMENT_BIASES}"
            )
            raise ValueError(msg)

        zone_biases = ("nearest_zone", "assigned_zone")
        if self.movement_bias in zone_biases and not self.mating_zone_centers:
            msg = (
                f"movement_bias is {self.movement_bias!r} but no mating zone "
                f"centres were provided; set pairing_method to 'mating_zone' "
                f"to initialise zones, or change movement_bias"
            )
            raise ValueError(msg)

    def _displacement(
        self,
        origin: FloatArray,
        target: FloatArray,
    ) -> FloatArray:
        """Return the planar displacement from one point to another.

        Parameters
        ----------
        origin
            Starting position.
        target
            Target position.

        Returns
        -------
            A two-element displacement, wrapped when boundaries are periodic.
        """
        if self.use_periodic_boundaries:
            return calculate_periodic_displacement(
                origin,
                target,
                self.world_size,
            )[:2]
        return np.asarray(target[:2] - origin[:2], dtype=float)

    def _distance(
        self,
        origin: FloatArray,
        target: FloatArray,
    ) -> float:
        """Return the planar distance between two points.

        Parameters
        ----------
        origin
            First position.
        target
            Second position.

        Returns
        -------
            The distance, wrapped when boundaries are periodic.
        """
        return float(np.linalg.norm(self._displacement(origin, target)))

    def _target_for(
        self,
        robot_idx: int,
        positions: list[FloatArray],
    ) -> FloatArray | None:
        """Choose the point this robot should head toward.

        Parameters
        ----------
        robot_idx
            Index of the robot within the population.
        positions
            Current core position of every robot.

        Returns
        -------
            The target position, or ``None`` when there is nothing to aim at.
        """
        current = positions[robot_idx]

        if self.movement_bias == "nearest_neighbor":
            nearest_idx: int | None = None
            nearest_distance = float("inf")
            for other_idx, other in enumerate(positions):
                if other_idx == robot_idx:
                    continue
                distance = self._distance(current, other)
                if MIN_NEIGHBOUR_DISTANCE < distance < nearest_distance:
                    nearest_distance = distance
                    nearest_idx = other_idx

            if (
                nearest_idx is not None
                and nearest_distance > NEAREST_NEIGHBOUR_DEAD_ZONE
            ):
                return positions[nearest_idx]
            return None

        if self.movement_bias == "nearest_zone":
            nearest_center: FloatArray | None = None
            nearest_distance = float("inf")
            for cx, cy in self.mating_zone_centers:
                zone_pos = np.array([cx, cy, current[2]], dtype=float)
                distance = self._distance(current, zone_pos)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_center = zone_pos
            return nearest_center

        if self.movement_bias == "assigned_zone":
            unique_id = self.population[robot_idx].unique_id
            if unique_id is None:
                return None
            zone_idx = self.assigned_zones.get(unique_id)
            if zone_idx is None or not (
                0 <= zone_idx < len(self.mating_zone_centers)
            ):
                return None
            cx, cy = self.mating_zone_centers[zone_idx]
            return np.array([cx, cy, current[2]], dtype=float)

        return None

    def directional_inputs(
        self,
        robot_idx: int,
        positions: list[FloatArray],
    ) -> FloatArray:
        """Compute the normalised heading fed to a robot's network.

        Parameters
        ----------
        robot_idx
            Index of the robot within the population.
        positions
            Current core position of every robot.

        Returns
        -------
            A unit heading toward the robot's target, or zeros when it has
            none.
        """
        target = self._target_for(robot_idx, positions)
        if target is None:
            return np.zeros(2, dtype=float)

        vector = self._displacement(positions[robot_idx], target)
        norm = float(np.linalg.norm(vector))
        if norm <= MIN_DIRECTION_NORM:
            return np.zeros(2, dtype=float)
        return vector / norm

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        positions: list[FloatArray],
    ) -> None:
        """Write one control step for every robot.

        Parameters
        ----------
        model
            The compiled model, used to bound the actuator index.
        data
            Simulation state, whose ``ctrl`` array is written in place.
        positions
            Current core position of every robot.
        """
        num_robots = min(len(self.substrates), len(positions))

        for robot_idx in range(num_robots):
            inputs = sensor_inputs(
                data,
                robot_idx,
                self.num_joints,
                self.directional_inputs(robot_idx, positions),
            )
            motor_outputs = self.substrates[robot_idx].activate(inputs)

            for j in range(self.num_joints):
                ctrl_idx = robot_idx * self.num_joints + j
                if ctrl_idx < model.nu and j < len(motor_outputs):
                    data.ctrl[ctrl_idx] = np.clip(
                        motor_outputs[j],
                        self.control_clip_min,
                        self.control_clip_max,
                    )


def run_mating_movement_phase(
    spawned: SpawnedPopulation,
    controller: MatingController,
    duration: float,
    *,
    use_periodic_boundaries: bool = False,
    world_size: tuple[float, float] = (10.0, 10.0),
    trajectory_samples: int = 100,
    recorder: GenerationRecorder | None = None,
) -> list[list[FloatArray]]:
    """Simulate the population moving toward their mating targets.

    Parameters
    ----------
    spawned
        The compiled shared world holding the whole population.
    controller
        The controller driving the robots.
    duration
        Length of the phase in simulated seconds.
    use_periodic_boundaries
        Whether robots that leave the world are wrapped back through the
        opposite edge.
    world_size
        World dimensions ``(width, height)``.
    trajectory_samples
        Approximate number of samples to record per trajectory.
    recorder
        Optional video and snapshot capture for this phase.

    Returns
    -------
        One trajectory of ``(x, y)`` samples per robot, in population order.
    """
    model = spawned.model
    data = spawned.data

    total_steps = max(1, int(duration / model.opt.timestep))
    sample_interval = max(1, total_steps // max(1, trajectory_samples))

    # Trajectory history is kept by ARIEL's tracker; the per-step positions
    # below are read separately because the controller needs them every step,
    # not only at the sampling interval.
    if spawned.tracker is not None:
        spawned.tracker.reset()
    spawned.record()

    positions = spawned.core_positions()

    for step in range(total_steps):
        controller.apply(model, data, positions)
        mujoco.mj_step(model, data)

        if use_periodic_boundaries:
            apply_periodic_boundaries_to_simulation(
                model,
                data,
                spawned.free_joint_qpos_adr,
                world_size,
            )

        positions = spawned.core_positions()

        if recorder is not None:
            recorder.capture(data, step, total_steps)

        if (step + 1) % sample_interval == 0:
            spawned.record()

    return spawned.tracked_trajectories()
