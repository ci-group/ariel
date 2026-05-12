"""
MorphologicalMeasures class for robot phenotype digraph analysis.

Mostly based on the revolve2 implementation:
https://github.com/ci-group/revolve2/blob/master/standards/revolve2/standards/morphological_measures.py
"""

from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# ruff ignore: PLR0904 (too-many-public-methods)
#   This class is a collection of measures, so many properties are expected


class MorphologicalMeasures:  # noqa: PLR0904
    """Modular robot morphological measures computed from a tree (blueprint).

    Works with a NetworkX directed graph representation of a robot.

    Inspired by the measures defined in:
        Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
        Search Space Analysis of Evolvable Robot Morphologies.
        EvoApplications 2018. LNCS vol 10784. Springer, Cham.
        https://doi.org/10.1007/978-3-319-77538-8_47

    Implementation adapted from the revolve2 implementation, with
    some measures renamed and others added for clarity and
    relevance to our experiments.

    Parameters
    ----------
    robot_graph : nx.DiGraph
        Directed graph of the robot phenotype.  Nodes require ``'type'`` and
        ``'rotation'`` attributes; edges require a ``'face'`` attribute.

    Raises
    ------
    ValueError
        If robot_graph is empty or contains more than one root node.

    Attributes
    ----------
    graph : nx.DiGraph
        The input robot graph.
    grid : NDArray
        3D object array placing each node at its integer grid position.
    symmetry_grid : NDArray
        Padded version of grid centred on the core, used for symmetry checks.
    core_grid_position : np.ndarray
        Integer (x, y, z) index of the core module inside grid.
    is_2d : bool
        ``True`` when every module's rotation is a multiple of 90° in the XY
        plane (i.e. the robot has no vertical extent beyond its flat layout).
    core_node : Any
        The unique root node of graph (in-degree == 0).
    modules : list
        All nodes in graph.
    bricks : list
        Nodes whose ``'type'`` is ``'BRICK'``.
    active_hinges : list
        Nodes whose ``'type'`` is ``'HINGE'``.
    core_is_filled : bool
        ``True`` if every allowed face of the core has a child module.
    filled_bricks : list
        Bricks that have all their allowed faces occupied by children.
    filled_active_hinges : list
        Active hinges that have their one allowed face occupied by a child.
    single_neighbour_modules : list
        Non-core modules connected to exactly one other module (leaf nodes).
    single_neighbour_bricks : list
        Bricks with no children (leaf bricks).
    double_neighbour_bricks : list
        Bricks with exactly one child (internal chain nodes).
    double_neighbour_active_hinges : list
        Active hinges with exactly one child (internal chain nodes).
    xy_symmetry : float
        Degree of symmetry across the XY plane (Z-axis mirror), in [0, 1].
    xz_symmetry : float
        Degree of symmetry across the XZ plane (Y-axis mirror), in [0, 1].
    yz_symmetry : float
        Degree of symmetry across the YZ plane (X-axis mirror), in [0, 1].
    """

    def __init__(self, robot_graph: nx.DiGraph) -> None:
        if robot_graph.number_of_nodes() == 0:
            msg = "Cannot analyze empty robot graph"
            raise ValueError(msg)

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_core_node(self) -> Any:
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        if len(roots) != 1:
            msg = f"Expected exactly one root node, found {len(roots)}"
            raise ValueError(msg)
        return roots[0]

    def _get_nodes_by_type(self, module_type: str) -> list:
        return [
            n for n in self.graph.nodes()
            if self.graph.nodes[n].get("type") == module_type
        ]

    def _calculate_is_2d(self) -> bool:
        valid = {"DEG_0", "DEG_90", "DEG_180", "DEG_270"}
        return all(
            self.graph.nodes[n].get("rotation", "DEG_0") in valid
            for n in self.graph.nodes()
        )

    def _get_node_type(self, node: Any) -> str:
        return self.graph.nodes[node].get("type", "UNKNOWN")

    def _get_allowed_faces(self, node: Any) -> list[str]:
        t = self._get_node_type(node)
        if t == "CORE":
            return ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"]
        if t == "BRICK":
            return ["FRONT", "RIGHT", "LEFT", "TOP", "BOTTOM"]
        if t == "HINGE":
            return ["FRONT"]
        return []

    def _get_node_connections(self, node: Any) -> list[str]:
        faces = []
        for successor in self.graph.successors(node):
            data = self.graph.get_edge_data(node, successor)
            if data and "face" in data:
                faces.append(data["face"])
        return faces

    def _count_neighbors(self, node: Any) -> int:
        return self.graph.in_degree(node) + self.graph.out_degree(node)

    def _graph_to_grid(
        self, robot_graph: nx.DiGraph,
    ) -> tuple[NDArray, np.ndarray]:
        if robot_graph.number_of_nodes() == 0:
            msg = "Cannot convert empty robot graph to grid"
            raise ValueError(msg)

        positions: dict[Any, np.ndarray] = {}
        core_node = self._find_core_node()
        self._calculate_graph_positions(core_node, positions, np.array([0, 0, 0]))

        if not positions:
            positions[core_node] = np.array([0, 0, 0])

        pos_array = np.array(list(positions.values()))
        min_pos = pos_array.min(axis=0)

        grid = np.full(pos_array.max(axis=0) - min_pos + 1, None, dtype=object)
        for node in robot_graph.nodes():
            grid[tuple(positions[node] - min_pos)] = node

        return grid, positions[core_node] - min_pos

    def _calculate_graph_positions(
        self, node: Any, positions: dict, pos: np.ndarray,
    ) -> None:
        positions[node] = pos.copy()
        face_directions = {
            "FRONT":  np.array([1, 0, 0]),
            "BACK":   np.array([-1, 0, 0]),
            "RIGHT":  np.array([0, 1, 0]),
            "LEFT":   np.array([0, -1, 0]),
            "TOP":    np.array([0, 0, 1]),
            "BOTTOM": np.array([0, 0, -1]),
        }
        for child in self.graph.successors(node):
            data = self.graph.get_edge_data(node, child)
            if data and "face" in data and data["face"] in face_directions:
                child_pos = pos + face_directions[data["face"]]
                if child not in positions:
                    self._calculate_graph_positions(child, positions, child_pos)

    def _calculate_core_is_filled(self) -> bool:
        return len(self._get_node_connections(self.core_node)) == len(
            self._get_allowed_faces(self.core_node),
        )

    def _calculate_filled_bricks(self) -> list:
        return [
            b for b in self.bricks
            if len(self._get_node_connections(b)) == len(self._get_allowed_faces(b))
        ]

    def _calculate_filled_active_hinges(self) -> list:
        return [
            h for h in self.active_hinges
            if len(self._get_node_connections(h)) == len(self._get_allowed_faces(h))
        ]

    def _calculate_single_neighbour_bricks(self) -> list:
        return [b for b in self.bricks if self.graph.out_degree(b) == 0]

    def _calculate_single_neighbour_modules(self) -> list:
        return [
            n for n in self.modules
            if self._get_node_type(n) != "CORE" and self._count_neighbors(n) == 1
        ]

    def _calculate_double_neighbour_bricks(self) -> list:
        return [b for b in self.bricks if self.graph.out_degree(b) == 1]

    def _calculate_double_neighbour_active_hinges(self) -> list:
        return [h for h in self.active_hinges if self.graph.out_degree(h) == 1]

    def _pad_grid(self) -> None:
        x, y, z = self.grid.shape
        xo, yo, zo = self.core_grid_position
        self.symmetry_grid = np.full((x + xo, y + yo, z + zo),
                                     None,
                                     dtype=object,
                                     )
        self.symmetry_grid[:x, :y, :z] = self.grid

    def _calculate_xy_symmetry(self) -> float:
        on_plane = off_sym = 0
        cz = self.core_grid_position[2]
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(self.bounding_box_width),
            range(1, (self.bounding_box_height - 1) // 2),
        ):
            if self.symmetry_grid[x, y, cz] is not None:
                on_plane += 1
            pos = self.symmetry_grid[x, y, cz + z]
            neg = self.symmetry_grid[x, y, cz - z]
            if pos is not None and neg is not None:
                if self._get_node_type(pos) == self._get_node_type(neg):
                    off_sym += 2
        diff = self.num_modules - on_plane
        return off_sym / diff if diff > 0 else 0.0

    def _calculate_xz_symmetry(self) -> float:
        on_plane = off_sym = 0
        cy = self.core_grid_position[1]
        for x, y, z in product(
            range(self.bounding_box_depth),
            range(1, (self.bounding_box_width - 1) // 2),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[x, cy, z] is not None:
                on_plane += 1
            pos = self.symmetry_grid[x, cy + y, z]
            neg = self.symmetry_grid[x, cy - y, z]
            if pos is not None and neg is not None:
                if self._get_node_type(pos) == self._get_node_type(neg):
                    off_sym += 2
        diff = self.num_modules - on_plane
        return off_sym / diff if diff > 0 else 0.0

    def _calculate_yz_symmetry(self) -> float:
        on_plane = off_sym = 0
        cx = self.core_grid_position[0]
        for x, y, z in product(
            range(1, (self.bounding_box_depth - 1) // 2),
            range(self.bounding_box_width),
            range(self.bounding_box_height),
        ):
            if self.symmetry_grid[cx, y, z] is not None:
                on_plane += 1
            pos = self.symmetry_grid[cx + x, y, z]
            neg = self.symmetry_grid[cx - x, y, z]
            if pos is not None and neg is not None:
                if self._get_node_type(pos) == self._get_node_type(neg):
                    off_sym += 2
        diff = self.num_modules - on_plane
        return off_sym / diff if diff > 0 else 0.0

    # ------------------------------------------------------------------
    # Bounding-box geometry
    # ------------------------------------------------------------------

    @property
    def bounding_box_depth(self) -> int:
        """Extent along the forward/backward (X) axis in module-grid units.

        Returns
        -------
        int
        """
        return self.grid.shape[0]

    @property
    def bounding_box_width(self) -> int:
        """Extent along the left/right (Y) axis in module-grid units.

        Returns
        -------
        int
        """
        return self.grid.shape[1]

    @property
    def bounding_box_height(self) -> int:
        """Extent along the up/down (Z) axis in module-grid units.

        Returns
        -------
        int
        """
        return self.grid.shape[2]

    @property
    def bounding_box_volume(self) -> int:
        """Product of all three bounding-box dimensions (m_area in the paper).

        Returns
        -------
        int
        """
        return (self.bounding_box_depth *
                self.bounding_box_width *
                self.bounding_box_height
                )

    # ------------------------------------------------------------------
    # Raw counts
    # ------------------------------------------------------------------

    @property
    def num_modules(self) -> int:
        """Total number of modules.

        Returns
        -------
        int
        """
        return len(self.modules)

    @property
    def num_bricks(self) -> int:
        """Number of BRICK-type modules.

        Returns
        -------
        int
        """
        return len(self.bricks)

    @property
    def num_active_hinges(self) -> int:
        """Number of HINGE-type modules.

        Returns
        -------
        int
        """
        return len(self.active_hinges)

    @property
    def num_joints(self) -> int:
        """Alias for ``num_active_hinges``.

        Returns
        -------
        int
        """
        return self.num_active_hinges

    @property
    def num_filled_bricks(self) -> int:
        """Number of bricks with every allowed face occupied.

        Returns
        -------
        int
        """
        return len(self.filled_bricks)

    @property
    def num_filled_active_hinges(self) -> int:
        """Number of active hinges with their allowed face occupied.

        Returns
        -------
        int
        """
        return len(self.filled_active_hinges)

    @property
    def num_filled_modules(self) -> int:
        """Total fully-saturated modules (filled bricks + filled hinges + filled core).

        Returns
        -------
        int
        """
        return (
            self.num_filled_bricks
            + self.num_filled_active_hinges
            + (1 if self.core_is_filled else 0)
        )

    @property
    def num_single_neighbour_modules(self) -> int:
        """Number of non-core leaf modules (connected to exactly one module).

        Returns
        -------
        int
        """
        return len(self.single_neighbour_modules)

    @property
    def num_double_neighbour_bricks(self) -> int:
        """Number of bricks connected to exactly two modules.

        Returns
        -------
        int
        """
        return len(self.double_neighbour_bricks)

    @property
    def num_double_neighbour_active_hinges(self) -> int:
        """Number of active hinges connected to exactly two modules.

        Returns
        -------
        int
        """
        return len(self.double_neighbour_active_hinges)

    # ------------------------------------------------------------------
    # Theoretical maxima
    # ------------------------------------------------------------------

    @property
    def max_potentionally_filled_core_and_bricks(self) -> int:
        """Upper bound on filled core+bricks given this module set (b_max).

        Returns
        -------
        int
        """
        return min(max(0, (self.num_modules - 2) // 3), 1 + self.num_bricks)

    @property
    def max_potential_single_neighbour_modules(self) -> int:
        """Upper bound on single-neighbour modules given this module set (l_max).

        Returns
        -------
        int
        """
        return self.num_modules - 1 - max(0, (self.num_modules - 3) // 3)

    @property
    def max_potential_joints(self) -> int:
        """Maximum possible joints if every inter-module connection were a hinge.

        Returns
        -------
        int
            ``num_modules - 1``, or 0 for a single-module robot.
        """
        return max(0, self.num_modules - 1)

    @property
    def potential_double_neighbour_bricks_and_active_hinges(self) -> int:
        """Upper bound on double-neighbour bricks+hinges given this module set (e_max).

        Returns
        -------
        int
        """
        return max(0, self.num_bricks + self.num_active_hinges - 1)

    # ------------------------------------------------------------------
    # Named proportions (intermediate, kept for backwards compatibility)
    # ------------------------------------------------------------------

    @property
    def filled_core_and_bricks_proportion(self) -> float:
        """Ratio of filled core+bricks to their theoretical maximum (branching).

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.max_potentionally_filled_core_and_bricks == 0:
            return 0.0
        return (
            len(self.filled_bricks) + (1 if self.core_is_filled else 0)
        ) / self.max_potentionally_filled_core_and_bricks

    @property
    def double_neighbour_brick_and_active_hinge_proportion(self) -> float:
        """Ratio of double-neighbour bricks+hinges to their theoretical maximum.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.potential_double_neighbour_bricks_and_active_hinges == 0:
            return 0.0
        return (
            self.num_double_neighbour_bricks + self.num_double_neighbour_active_hinges
        ) / self.potential_double_neighbour_bricks_and_active_hinges

    @property
    def bounding_box_volume_coverage(self) -> float:
        """Fraction of the bounding-box volume occupied by modules.

        Returns
        -------
        float
            Value in (0, 1].
        """
        return self.num_modules / self.bounding_box_volume

    @property
    def branching(self) -> float:
        """Alias for ``filled_core_and_bricks_proportion``.

        Returns
        -------
        float
        """
        return self.filled_core_and_bricks_proportion

    @property
    def limbs(self) -> float:
        """Fraction of non-core leaf modules, normalised by l_max.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.max_potential_single_neighbour_modules == 0:
            return 0.0
        return (
            self.num_single_neighbour_modules / self.max_potential_single_neighbour_modules
        )

    @property
    def length_of_limbs(self) -> float:
        """Alias for ``double_neighbour_brick_and_active_hinge_proportion``.

        Returns
        -------
        float
        """
        return self.double_neighbour_brick_and_active_hinge_proportion

    @property
    def coverage(self) -> float:
        """Alias for ``bounding_box_volume_coverage``.

        Returns
        -------
        float
        """
        return self.bounding_box_volume_coverage

    @property
    def proportion_2d(self) -> float:
        """Proportion measure for 2D robots only.

        Returns
        -------
        float
            min(depth, width) / max(depth, width), in (0, 1].
        """
        return min(self.bounding_box_depth, self.bounding_box_width) / max(
            self.bounding_box_depth, self.bounding_box_width,
        )

    @property
    def proportion(self) -> float:
        """Ground-plane footprint proportion; delegates to ``P``.

        Returns
        -------
        float
            min(depth, width) / max(depth, width), in (0, 1].
        """
        return self.P

    @property
    def symmetry(self) -> float:
        """Best-of-three plane symmetry score.

        Returns
        -------
        float
            max(xy_symmetry, xz_symmetry, yz_symmetry).
        """
        return max(self.xy_symmetry, self.xz_symmetry, self.yz_symmetry)

    @property
    def module_diversity(self) -> float:
        """Fraction of BRICK/HINGE edges that alternate module type.

        Returns
        -------
        float
            Value in [0, 1]; 0.0 if no BRICK/HINGE edge exists.
        """
        alt = total = 0
        for u, v in self.graph.edges():
            t1, t2 = self._get_node_type(u), self._get_node_type(v)
            if t1 in {"BRICK", "HINGE"} and t2 in {"BRICK", "HINGE"}:
                total += 1
                if t1 != t2:
                    alt += 1
        return alt / total if total > 0 else 0.0

    @property
    def joints(self) -> float:
        """Joint ratio J = num_joints / max_potential_joints.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.max_potential_joints == 0:
            return 0.0
        return self.num_joints / self.max_potential_joints

    @property
    def size(self) -> float:
        """Size = num_modules / bounding_box_volume.

        Returns
        -------
        float
            Value in (0, 1].
        """
        # TODO: verify that bounding_box_volume is the right normaliser here
        if self.bounding_box_volume == 0:
            return 0.0
        return self.num_modules / self.bounding_box_volume

    @property
    def brick_cluster_ratio(self) -> float:
        """Fraction of brick-involved edges that connect two bricks.

        Complements ``D``: ``D`` measures global alternation; this penalises
        dense brick-to-brick clusters.

        Returns
        -------
        float
            0.0 = every brick connects only to hinges.
            1.0 = every brick connects only to other bricks.
        """
        brick_set = set(self.bricks)
        brick_total = brick_brick = 0
        for u, v in self.graph.edges():
            u_b, v_b = u in brick_set, v in brick_set
            if u_b or v_b:
                brick_total += 1
                if u_b and v_b:
                    brick_brick += 1
        return brick_brick / brick_total if brick_total > 0 else 0.0

    # ------------------------------------------------------------------
    # Paper-letter aliases
    # ------------------------------------------------------------------

    @property
    def B(self) -> float:
        """Branching B = b / b_max."""
        return self.branching

    @property
    def L(self) -> float:
        """Limbs L = l / l_max."""
        return self.limbs

    @property
    def E(self) -> float:
        """Extensiveness (length of limbs) E = e / e_max."""
        return self.length_of_limbs

    @property
    def C(self) -> float:
        """Coverage C = num_modules / bounding_box_volume."""
        return self.coverage

    @property
    def J(self) -> float:
        """Joints J = j / j_max."""
        return self.joints

    @property
    def S(self) -> float:
        """Symmetry S = max(xy, xz, yz symmetry)."""
        return self.symmetry

    @property
    def D(self) -> float:
        """Module diversity D = alternating BRICK/HINGE fraction."""
        return self.module_diversity

    @property
    def P(self) -> float:
        """Proportion P of the ground-plane (depth x width) footprint.

        Uses depth and width only, valid for both 2D and 3D robots.

        Returns
        -------
        float
            min(depth, width) / max(depth, width), in (0, 1].
        """
        d, w = self.bounding_box_depth, self.bounding_box_width
        return min(d, w) / max(d, w) if max(d, w) > 0 else 0.0
