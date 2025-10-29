"""
MorphologicalMeasures class for robot phenotype digraph analysis.
Mostly based on the revolve implementation: https://github.com/ci-group/revolve2/blob/master/standards/revolve2/standards/morphological_measures.py
"""

from itertools import product
from typing import Generic, TypeVar, Any

import numpy as np
from numpy.typing import NDArray
import networkx as nx

import ariel.body_phenotypes.robogen_lite.config as config

TModule = TypeVar("TModule", bound=np.generic)


class MorphologicalMeasures(Generic[TModule]):
    """
    Modular robot morphological measures for robot phenotype digraph.

    Works with a NetworkX directed graph representation of a robot.
    Only works for robot with only right angle module rotations (90 degrees).
    Some measures only work for 2d robots, which is noted in their docstring.

    The measures are based on the following paper:
    Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
    Search Space Analysis of Evolvable Robot Morphologies.
    In: Sim, K., Kaufmann, P. (eds) Applications of Evolutionary Computation.
    EvoApplications 2018. Lecture Notes in Computer Science(), vol 10784. Springer, Cham.
    https://doi.org/10.1007/978-3-319-77538-8_47
    """

    """Represents the modules of a body in a 3D tensor."""
    grid: NDArray[TModule]
    symmetry_grid: NDArray[TModule]
    """Position of the core in 'body_as_grid'."""
    core_grid_position: np.ndarray

    """If the robot is two dimensional, i.e. all module rotations are 0 degrees."""
    is_2d: bool

    """The robot graph structure."""
    graph: nx.DiGraph
    core_node: Any
    modules: list[Any]
    bricks: list[Any]
    active_hinges: list[Any]

    """If all slots of the core are filled with other modules."""
    core_is_filled: bool

    """Bricks which have all slots filled with other modules."""
    filled_bricks: list[Any]

    """Active hinges which have all slots filled with other modules."""
    filled_active_hinges: list[Any]

    """
    Modules that only connect to one other module.

    This includes children and parents.
    """
    single_neighbour_modules: list[Any]

    """
    Bricks that are only connected to one other module.

    Both children and parent are counted.
    """
    single_neighbour_bricks: list[Any]

    """
    Bricks that are connected to exactly two other modules.

    Both children and parent are counted.
    """
    double_neighbour_bricks: list[Any]

    """
    Active hinges that are connected to exactly two other modules.

    Both children and parent are counted.
    """
    double_neighbour_active_hinges: list[Any]

    """
    X/Y-plane symmetry according to the paper but in 3D.

    X-axis is defined as forward/backward for the core module
    Y-axis is defined as left/right for the core module.
    """
    xy_symmetry: float

    """
    X/Z-plane symmetry according to the paper but in 3D.

    X-axis is defined as forward/backward for the core module
    Z-axis is defined as up/down for the core module.
    """
    xz_symmetry: float

    """
    Y/Z-plane symmetry according to the paper but in 3D.

    Y-axis is defined as left/right for the core module.
    Z-axis is defined as up/down for the core module.
    """
    yz_symmetry: float

    def __init__(self, robot_graph: nx.DiGraph) -> None:
        """
        Initialize this object.

        :param robot_graph: The NetworkX directed graph representing the robot phenotype.
                           Expected to have node attributes 'type' and 'rotation'.
                           Expected to have edge attributes 'face'.
        """
        if robot_graph.number_of_nodes() == 0:
            raise ValueError("Cannot analyze empty robot graph")

        self.graph = robot_graph
        self.grid, self.core_grid_position = self._graph_to_grid(robot_graph)
        self.core_node = self._find_core_node()
        self.is_2d = self._calculate_is_2d()
        self.modules = list(robot_graph.nodes())
        self.bricks = self._get_nodes_by_type("BRICK")
        self.active_hinges = self._get_nodes_by_type("HINGE")
        self.core_is_filled = self._calculate_core_is_filled()
        self.filled_bricks = self._calculate_filled_bricks()
        self.filled_active_hinges = self._calculate_filled_active_hinges()
        self.single_neighbour_bricks = self._calculate_single_neighbour_bricks()
        self.single_neighbour_modules = self._calculate_single_neighbour_modules()
        self.double_neighbour_bricks = self._calculate_double_neighbour_bricks()
        self.double_neighbour_active_hinges = (
            self._calculate_double_neighbour_active_hinges()
        )

        self._pad_grid()
        self.xy_symmetry = self._calculate_xy_symmetry()
        self.xz_symmetry = self._calculate_xz_symmetry()
        self.yz_symmetry = self._calculate_yz_symmetry()

    def _find_core_node(self) -> Any:
        """Find the core node (root of the tree) in the graph."""
        # Find node with no predecessors (root)
        roots = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root node, found {len(roots)}")
        return roots[0]

    def _get_nodes_by_type(self, module_type: str) -> list[Any]:
        """Get all nodes of a specific module type."""
        return [node for node in self.graph.nodes()
                if self.graph.nodes[node].get('type') == module_type]

    def _calculate_is_2d(self) -> bool:
        """Check if all modules use only 90-degree rotations."""
        valid_rotations = {"DEG_0", "DEG_90", "DEG_180", "DEG_270"}
        return all(
            self.graph.nodes[node].get('rotation', 'DEG_0') in valid_rotations
            for node in self.graph.nodes()
        )

    def _get_node_type(self, node: Any) -> str:
        """Get the module type of a node."""
        return self.graph.nodes[node].get('type', 'UNKNOWN')

    def _get_allowed_faces(self, node: Any) -> list[str]:
        """Get allowed faces for a node based on its type."""
        module_type = self._get_node_type(node)
        if module_type == "CORE":
            return ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"]
        elif module_type == "BRICK":
            return ["FRONT", "RIGHT", "LEFT", "TOP", "BOTTOM"]
        elif module_type == "HINGE":
            return ["FRONT"]
        else:
            return []

    def _get_node_connections(self, node: Any) -> list[str]:
        """Get the faces that are connected for a node."""
        connected_faces = []
        # Check outgoing edges (children)
        for successor in self.graph.successors(node):
            edge_data = self.graph.get_edge_data(node, successor)
            if edge_data and 'face' in edge_data:
                connected_faces.append(edge_data['face'])
        return connected_faces

    def _count_neighbors(self, node: Any) -> int:
        """Count total neighbors (predecessors + successors)."""
        return self.graph.in_degree(node) + self.graph.out_degree(node)

    def _graph_to_grid(self, robot_graph: nx.DiGraph) -> tuple[NDArray[TModule], np.ndarray]:
        """Convert robot graph to 3D grid representation."""
        if robot_graph.number_of_nodes() == 0:
            raise ValueError("Cannot convert empty robot graph to grid")

        # Calculate positions of all nodes relative to core
        positions = {}
        core_node = self._find_core_node()
        self._calculate_graph_positions(core_node, positions, np.array([0, 0, 0]))

        # Find bounds
        if not positions:
            # Single core only
            positions[core_node] = np.array([0, 0, 0])

        pos_array = np.array(list(positions.values()))
        min_pos = pos_array.min(axis=0)
        max_pos = pos_array.max(axis=0)

        # Create grid with proper size
        grid_size = max_pos - min_pos + 1
        grid = np.full(grid_size, None, dtype=object)

        # Place nodes in grid
        core_pos = positions[core_node] - min_pos
        for node in robot_graph.nodes():
            node_pos = positions[node] - min_pos
            grid[tuple(node_pos)] = node

        return grid, core_pos

    def _calculate_graph_positions(self, node: Any, positions: dict, pos: np.ndarray) -> None:
        """Recursively calculate 3D positions of all nodes in the graph."""
        positions[node] = pos.copy()

        # Define face direction vectors (assuming standard orientation)
        face_directions = {
            "FRONT": np.array([1, 0, 0]),
            "BACK": np.array([-1, 0, 0]),
            "RIGHT": np.array([0, 1, 0]),
            "LEFT": np.array([0, -1, 0]),
            "TOP": np.array([0, 0, 1]),
            "BOTTOM": np.array([0, 0, -1]),
        }

        # Process children (successors in the graph)
        for child in self.graph.successors(node):
            edge_data = self.graph.get_edge_data(node, child)
            if edge_data and 'face' in edge_data:
                face = edge_data['face']
                if face in face_directions:
                    child_pos = pos + face_directions[face]
                    if child not in positions:  # Avoid cycles
                        self._calculate_graph_positions(child, positions, child_pos)

    def _calculate_core_is_filled(self) -> bool:
        """Check if the core has all its allowed faces filled."""
        allowed_faces = self._get_allowed_faces(self.core_node)
        connected_faces = self._get_node_connections(self.core_node)
        return len(connected_faces) == len(allowed_faces)

    def _calculate_filled_bricks(self) -> list[Any]:
        """Get bricks that have all their allowed faces filled."""
        return [
            brick
            for brick in self.bricks
            if len(self._get_node_connections(brick)) == len(self._get_allowed_faces(brick))
        ]

    def _calculate_filled_active_hinges(self) -> list[Any]:
        """Get active hinges that have all their allowed faces filled."""
        return [
            hinge
            for hinge in self.active_hinges
            if len(self._get_node_connections(hinge)) == len(self._get_allowed_faces(hinge))
        ]

    def _calculate_single_neighbour_bricks(self) -> list[Any]:
        """Get bricks that have no children (leaf nodes)."""
        return [
            brick
            for brick in self.bricks
            if self.graph.out_degree(brick) == 0
        ]

    def _calculate_single_neighbour_modules(self) -> list[Any]:
        """Get non-core modules that have only one neighbor (leaf nodes)."""
        non_core_modules = [node for node in self.modules if self._get_node_type(node) != "CORE"]
        return [
            module
            for module in non_core_modules
            if self._count_neighbors(module) == 1
        ]

    def _calculate_double_neighbour_bricks(self) -> list[Any]:
        """Get bricks that have exactly one child (connecting two modules)."""
        return [
            brick
            for brick in self.bricks
            if self.graph.out_degree(brick) == 1
        ]

    def _calculate_double_neighbour_active_hinges(self) -> list[Any]:
        """Get active hinges that have exactly one child (connecting two modules)."""
        return [
            hinge
            for hinge in self.active_hinges
            if self.graph.out_degree(hinge) == 1
        ]

    def _pad_grid(self) -> None:
        x, y, z = self.grid.shape
        xoffs, yoffs, zoffs = self.core_grid_position
        self.symmetry_grid = np.empty(
            shape=(x + xoffs, y + yoffs, z + zoffs), dtype=object
        )
        self.symmetry_grid.fill(None)
        self.symmetry_grid[:x, :y, :z] = self.grid

    def _calculate_xy_symmetry(self) -> float:
        """Calculate XY-plane symmetry."""
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(self.bounding_box_width),
            range(1, (self.bounding_box_height - 1) // 2),
        ):
            if self.symmetry_grid[x, y, self.core_grid_position[2]] is not None:
                num_along_plane += 1
            pos_z = self.symmetry_grid[x, y, self.core_grid_position[2] + z]
            neg_z = self.symmetry_grid[x, y, self.core_grid_position[2] - z]
            if pos_z is not None and neg_z is not None:
                # Check if module types match
                if self._get_node_type(pos_z) == self._get_node_type(neg_z):
                    num_symmetrical += 2

        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else 0.0

    def _calculate_xz_symmetry(self) -> float:
        """Calculate XZ-plane symmetry."""
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(1, (self.bounding_box_width - 1) // 2),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[x, self.core_grid_position[1], z] is not None:
                num_along_plane += 1
            pos_y = self.symmetry_grid[x, self.core_grid_position[1] + y, z]
            neg_y = self.symmetry_grid[x, self.core_grid_position[1] - y, z]
            if pos_y is not None and neg_y is not None:
                # Check if module types match
                if self._get_node_type(pos_y) == self._get_node_type(neg_y):
                    num_symmetrical += 2
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else 0.0

    def _calculate_yz_symmetry(self) -> float:
        """Calculate YZ-plane symmetry."""
        num_along_plane = 0
        num_symmetrical = 0
        for x, y, z in product(
            range(1, (self.bounding_box_depth - 1) // 2),
            range(self.bounding_box_width),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[self.core_grid_position[0], y, z] is not None:
                num_along_plane += 1
            pos_x = self.symmetry_grid[self.core_grid_position[0] + x, y, z]
            neg_x = self.symmetry_grid[self.core_grid_position[0] - x, y, z]
            if pos_x is not None and neg_x is not None:
                # Check if module types match
                if self._get_node_type(pos_x) == self._get_node_type(neg_x):
                    num_symmetrical += 2
        difference = self.num_modules - num_along_plane
        return num_symmetrical / difference if difference > 0.0 else 0.0

    @property
    def bounding_box_depth(self) -> int:
        """
        Get the depth of the bounding box around the body.

        Forward/backward axis for the core module.

        :returns: The depth.
        """
        return self.grid.shape[0]

    @property
    def bounding_box_width(self) -> int:
        """
        Get the width of the bounding box around the body.

        Right/left axis for the core module.

        :returns: The width.
        """
        return self.grid.shape[1]

    @property
    def bounding_box_height(self) -> int:
        """
        Get the height of the bounding box around the body.

        Up/down axis for the core module.

        :returns: The height.
        """
        return self.grid.shape[2]

    @property
    def num_modules(self) -> int:
        """
        Get the number of modules.

        :returns: The number of modules.
        """
        return len(self.modules)

    @property
    def num_bricks(self) -> int:
        """
        Get the number of bricks.

        :returns: The number of bricks.
        """
        return len(self.bricks)

    @property
    def num_active_hinges(self) -> int:
        """
        Get the number of active hinges.

        :returns: The number of active hinges.
        """
        return len(self.active_hinges)

    @property
    def num_filled_bricks(self) -> int:
        """
        Get the number of bricks which have all slots filled with other modules.

        :returns: The number of bricks.
        """
        return len(self.filled_bricks)

    @property
    def num_filled_active_hinges(self) -> int:
        """
        Get the number of bricks which have all slots filled with other modules.

        :returns: The number of bricks.
        """
        return len(self.filled_active_hinges)

    @property
    def num_filled_modules(self) -> int:
        """
        Get the number of modules which have all slots filled with other modules, including the core.

        :returns: The number of modules.
        """
        return (
            self.num_filled_bricks
            + self.num_filled_active_hinges
            + (1 if self.core_is_filled else 0)
        )

    @property
    def max_potentionally_filled_core_and_bricks(self) -> int:
        """
        Get the maximum number of core and bricks that could potentially be filled with this set of modules if rearranged in an optimal way.

        This calculates 'b_max' from the paper.

        :returns: The calculated number.
        """
        pot_max_filled = max(0, (self.num_modules - 2) // 3)
        pot_max_filled = min(pot_max_filled, 1 + self.num_bricks)
        return pot_max_filled

    @property
    def filled_core_and_bricks_proportion(self) -> float:
        """
        Get the ratio between filled cores and bricks and how many that potentially could have been if this set of modules was rearranged in an optimal way.

        This calculates 'branching' from the paper.

        :returns: The proportion.
        """
        if self.max_potentionally_filled_core_and_bricks == 0:
            return 0.0

        return (
            len(self.filled_bricks) + (1 if self.core_is_filled else 0)
        ) / self.max_potentionally_filled_core_and_bricks

    @property
    def num_single_neighbour_modules(self) -> int:
        """
        Get the number of bricks that are only connected to one other module.

        Both children and parent are counted.

        :returns: The number of bricks.
        """
        return len(self.single_neighbour_modules)

    @property
    def max_potential_single_neighbour_modules(self) -> int:
        """
        Get the maximum number of bricks that could potentially have only one neighbour if this set of modules was rearranged in an optimal way.

        This calculates "l_max" from the paper.

        :returns: The calculated number.
        """
        return self.num_modules - 1 - max(0, (self.num_modules - 3) // 3)

    @property
    def num_double_neighbour_bricks(self) -> int:
        """
        Get the number of bricks that are connected to exactly two other modules.

        Both children and parent are counted.

        :returns: The number of bricks.
        """
        return len(self.double_neighbour_bricks)

    @property
    def num_double_neighbour_active_hinges(self) -> int:
        """
        Get the number of active hinges that are connected to exactly two other modules.

        Both children and parent are counted.

        :returns: The number of active hinges.
        """
        return len(self.double_neighbour_active_hinges)

    @property
    def potential_double_neighbour_bricks_and_active_hinges(self) -> int:
        """
        Get the maximum number of bricks and active hinges that could potentially have exactly two neighbours if this set of modules was rearranged in an optimal way.

        This calculates e_max from the paper.

        :returns: The calculated number.
        """
        return max(0, self.num_bricks + self.num_active_hinges - 1)

    @property
    def double_neighbour_brick_and_active_hinge_proportion(self) -> float:
        """
        Get the ratio between the number of bricks and active hinges with exactly two neighbours and how many that could potentially have been if this set of modules was rearranged in an optimal way.

        This calculate length of limbs proportion(extensiveness) from the paper.

        :returns: The proportion.
        """
        if self.potential_double_neighbour_bricks_and_active_hinges == 0:
            return 0.0

        return (
            self.num_double_neighbour_bricks + self.num_double_neighbour_active_hinges
        ) / self.potential_double_neighbour_bricks_and_active_hinges

    @property
    def bounding_box_volume(self) -> int:
        """
        Get the volume of the bounding box.

        This calculates m_area from the paper.

        :returns: The volume.
        """
        return (
            self.bounding_box_width * self.bounding_box_height * self.bounding_box_depth
        )

    @property
    def bounding_box_volume_coverage(self) -> float:
        """
        Get the proportion of the bounding box that is filled with modules.

        This calculates 'coverage' from the paper.

        :returns: The proportion.
        """
        return self.num_modules / self.bounding_box_volume

    @property
    def branching(self) -> float:
        """
        Get the 'branching' measurement from the paper.

        Alias for filled_core_and_bricks_proportion.

        :returns: Branching measurement.
        """
        return self.filled_core_and_bricks_proportion

    @property
    def limbs(self) -> float:
        """
        Get the 'limbs' measurement from the paper.

        Alias for single_neighbour_brick_proportion.

        :returns: Limbs measurement.
        """
        if self.max_potential_single_neighbour_modules == 0:
            return 0.0
        return (
            self.num_single_neighbour_modules
            / self.max_potential_single_neighbour_modules
        )

    @property
    def length_of_limbs(self) -> float:
        """
        Get the 'length of limbs' measurement from the paper.

        Alias for double_neighbour_brick_and_active_hinge_proportion.

        :returns: Length of limbs measurement.
        """
        return self.double_neighbour_brick_and_active_hinge_proportion

    @property
    def coverage(self) -> float:
        """
        Get the 'coverage' measurement from the paper.

        Alias for bounding_box_volume_coverage.

        :returns: Coverage measurement.
        """
        return self.bounding_box_volume_coverage

    @property
    def proportion_2d(self) -> float:
        """
        Get the 'proportion' measurement from the paper.

        Only for 2d robots.

        :returns: Proportion measurement.
        """
        assert self.is_2d

        return min(self.bounding_box_depth, self.bounding_box_width) / max(
            self.bounding_box_depth, self.bounding_box_width
        )

    @property
    def symmetry(self) -> float:
        """
        Get the 'symmetry' measurement from the paper, but extended to 3d.

        :returns: Symmetry measurement.
        """
        return max(self.xy_symmetry, self.xz_symmetry, self.yz_symmetry)

    @property
    def num_joints(self) -> int:
        """Number of joints (active hinges)."""
        return self.num_active_hinges

    @property
    def max_potential_joints(self) -> int:
        """Maximum possible joints (if every connection were a hinge)."""
        return max(0, self.num_modules - 1)

    @property
    def joints(self) -> float:
        """Get the 'number of joints' measurement J = j / j_max."""
        if self.max_potential_joints == 0:
            return 0.0
        return self.num_joints / self.max_potential_joints

    @property
    def size(self) -> float:
    #TODO check if m_max is fine like this!!
        """Size S = m / m_max (proportion of occupied volume).

        m = number of modules
        m_max = bounding box volume (max possible occupancy)

        Equivalent to 'coverage' in Miras et al. (2018) if volume is used as reference.
        """
        if self.bounding_box_volume == 0:
            return 0.0
        return self.num_modules / self.bounding_box_volume

    @property
    def proportion(self) -> float:
        """Proportion P = p_s / p_l (only valid for 2D morphologies)."""
        return self.proportion_2d

    @property
    def B(self) -> float:
        """Branching B = b / b_max."""
        return self.branching

    @property
    def L(self) -> float:
        """Length of limbs L = e / e_max."""
        return self.limbs

    @property
    def S(self) -> float:
        """Symmetry S = s."""
        return self.symmetry

    @property
    def C(self) -> float:
        """Coverage C = c."""
        return self.coverage

    @property
    def J(self) -> float:
        """Joints J = j / j_max."""
        return self.joints

    @property
    def E(self) -> float:
        """Extensiveness E = e / e_max."""
        return self.length_of_limbs

    @property
    def P(self) -> float:
        """Proportion P = p_s / p_l (only valid for 2D morphologies)."""
        if self.is_2d:
            return self.proportion_2d
        else:
            return 0.0  # Return 0 for 3D robots where proportion is not defined
