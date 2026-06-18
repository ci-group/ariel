"""CPPN-NEAT indirect-encoding genome handler for drone evolution."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base import GenomeHandler
from .cppn.network import (
    ActivationFunction,
    CPPNNetwork,
    ConnectionGene,
    NodeGene,
    NodeType,
)
from .cppn.innovation import InnovationCounter
from .cppn.evaluation import evaluate_cppn
from .cppn.segment_decoder import decode_cppn_to_phenotype
from .cppn.mutations import mutate_cppn
from .cppn.crossover import crossover_cppn
from .cppn.compatibility import cppn_compatibility_distance
from .operators import SphericalRepairOperator, RepairConfig


# Number of CPPN input nodes (segment_normalized, bias)
_N_INPUTS = 2
# Number of CPPN output nodes
_N_OUTPUTS = 7
# Labels for the output nodes
_OUTPUT_LABELS = [
    "arm_present", "magnitude", "arm_yaw", "arm_pitch",
    "motor_yaw", "motor_pitch", "direction",
]


class CPPNNeatDroneGenomeHandler(GenomeHandler):
    """Genome handler that uses a CPPN-NEAT indirect encoding.

    The genome is a :class:`CPPNNetwork` (directed acyclic graph).  The CPPN
    takes a normalised segment index and a bias as input and produces 7 outputs
    used to decide arm placement and parameters.
    """

    # Shared across the entire population so that structural mutations within
    # the same generation receive matching innovation numbers.
    _innovation_counter: InnovationCounter = InnovationCounter()

    def __init__(
        self,
        genome: Optional[CPPNNetwork] = None,
        num_segments: int = 8,
        min_max_narms: Optional[Tuple[int, int]] = None,
        parameter_limits: Optional[npt.NDArray[Any]] = None,
        # Mutation probabilities
        prob_add_node: float = 0.03,
        prob_add_connection: float = 0.05,
        prob_remove_node: float = 0.01,
        prob_remove_connection: float = 0.02,
        prob_mutate_weights: float = 0.80,
        prob_mutate_activation: float = 0.05,
        prob_toggle_connection: float = 0.02,
        # Initial topology complexity
        initial_hidden_nodes: int = 0,
        init_topology: str = "empty",  # "empty" or "seeded"
        # Weight / bias mutation parameters
        weight_perturb_std: float = 0.5,
        weight_replace_prob: float = 0.1,
        weight_range: float = 3.0,
        bias_perturb_std: float = 0.3,
        bias_replace_prob: float = 0.1,
        bias_range: float = 3.0,
        # Repair
        repair: bool = False,
        enable_collision_repair: bool = False,
        propeller_radius: float = 0.0508 / 2,
        inner_boundary_radius: float = 0.0055,
        outer_boundary_radius: float = 0.11,
        max_repair_iterations: int = 100,
        repair_step_size: float = 1.0,
        propeller_tolerance: float = 0.1,
        # RNG
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        # --- Do NOT call super().__init__() ---
        self.fitness: float | None = None
        self.rng = rng if rng is not None else np.random.default_rng()

        self.num_segments = num_segments

        if min_max_narms is None:
            self.min_narms, self.max_narms = 6, 6
        else:
            self.min_narms, self.max_narms = min_max_narms

        if parameter_limits is None:
            self.parameter_limits = np.array([
                [0.055, 0.17],           # magnitude
                [-np.pi, np.pi],         # arm yaw (azimuth)
                [-np.pi / 2, np.pi / 2], # arm pitch (elevation)
                [-np.pi, np.pi],         # motor pitch
                [-np.pi, np.pi],         # motor yaw
                [0, 1],                  # direction
            ])
        else:
            self.parameter_limits = np.asarray(parameter_limits)

        # Initial topology
        self.initial_hidden_nodes = initial_hidden_nodes
        self.init_topology = init_topology

        # Mutation hyperparameters
        self.prob_add_node = prob_add_node
        self.prob_add_connection = prob_add_connection
        self.prob_remove_node = prob_remove_node
        self.prob_remove_connection = prob_remove_connection
        self.prob_mutate_weights = prob_mutate_weights
        self.prob_mutate_activation = prob_mutate_activation
        self.prob_toggle_connection = prob_toggle_connection
        self.weight_perturb_std = weight_perturb_std
        self.weight_replace_prob = weight_replace_prob
        self.weight_range = weight_range
        self.bias_perturb_std = bias_perturb_std
        self.bias_replace_prob = bias_replace_prob
        self.bias_range = bias_range

        # Repair settings
        self.repair_enabled = repair
        self.enable_collision_repair = enable_collision_repair
        self.propeller_radius = propeller_radius
        self.inner_boundary_radius = inner_boundary_radius
        self.outer_boundary_radius = outer_boundary_radius
        self.max_repair_iterations = max_repair_iterations
        self.repair_step_size = repair_step_size
        self.propeller_tolerance = propeller_tolerance

        self._setup_repair_operator()

        # Genome
        if genome is None:
            self.genome: CPPNNetwork = self._generate_random_genome()
        else:
            self.genome = genome.copy()

    # ------------------------------------------------------------------
    # Repair operator setup
    # ------------------------------------------------------------------

    def _setup_repair_operator(self) -> None:
        repair_config = RepairConfig(
            apply_symmetry=False,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
        )
        self.repair_operator = SphericalRepairOperator(
            config=repair_config,
            min_narms=self.min_narms,
            max_narms=self.max_narms,
            parameter_limits=self.parameter_limits,
            symmetry_operator=None,
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # Random genome generation
    # ------------------------------------------------------------------

    # Activation functions that introduce useful spatial variation
    _HIDDEN_ACTIVATIONS = [
        ActivationFunction.SIN,
        ActivationFunction.COS,
        ActivationFunction.GAUSSIAN,
        ActivationFunction.TANH,
        ActivationFunction.ABS,
    ]

    # Activation functions used when seeding initial hidden nodes
    _SEED_ACTIVATIONS = [
        ActivationFunction.SIGMOID,
        ActivationFunction.TANH,
        ActivationFunction.GAUSSIAN,
    ]

    def _generate_random_genome(self) -> CPPNNetwork:
        """Create a CPPN with input and output nodes.

        If ``init_topology == "seeded"``, adds 2–5 hidden nodes with
        sigmoid/tanh/gaussian activations and ~10–20 random feed-forward
        connections.  Otherwise the network starts empty (classic NEAT
        complexification).
        """
        net = CPPNNetwork()

        # --- Input nodes ---
        for i in range(_N_INPUTS):
            label = "seg_normalized" if i == 0 else "bias"
            net.nodes[i] = NodeGene(
                node_id=i,
                node_type=NodeType.INPUT,
                activation=ActivationFunction.IDENTITY,
                bias=0.0,
                input_label=label,
            )

        # --- Output nodes (zero bias so all initial individuals are identical) ---
        for j in range(_N_OUTPUTS):
            nid = _N_INPUTS + j
            net.nodes[nid] = NodeGene(
                node_id=nid,
                node_type=NodeType.OUTPUT,
                activation=ActivationFunction.SIN,
                bias=0.0,
                output_index=j,
            )

        net.next_node_id = _N_INPUTS + _N_OUTPUTS

        if self.init_topology == "seeded":
            self._seed_topology(net, _N_INPUTS, _N_OUTPUTS)

        return net

    def _seed_topology(
        self, net: CPPNNetwork, n_inputs: int, n_outputs: int,
    ) -> None:
        """Add 2–5 hidden nodes and ~10–20 random feed-forward connections."""
        n_hidden = int(self.rng.integers(2, 6))

        input_ids = list(range(n_inputs))
        output_ids = list(range(n_inputs, n_inputs + n_outputs))
        hidden_ids = []

        for _ in range(n_hidden):
            nid = net.next_node_id
            net.next_node_id += 1
            activation = self.rng.choice(self._SEED_ACTIVATIONS)
            net.nodes[nid] = NodeGene(
                node_id=nid,
                node_type=NodeType.HIDDEN,
                activation=activation,
                bias=float(self.rng.uniform(-1.0, 1.0)),
            )
            hidden_ids.append(nid)

        # Valid feed-forward connections (input→hidden, input→output,
        # hidden→output, hidden→higher-hidden to keep DAG)
        possible = []
        for src in input_ids:
            for tgt in hidden_ids + output_ids:
                possible.append((src, tgt))
        for src in hidden_ids:
            for tgt in output_ids:
                possible.append((src, tgt))
        for i, src in enumerate(hidden_ids):
            for tgt in hidden_ids[i + 1:]:
                possible.append((src, tgt))

        n_target = int(self.rng.integers(10, 21))
        n_conns = min(n_target, len(possible))
        chosen = self.rng.choice(len(possible), size=n_conns, replace=False)

        for idx in chosen:
            src, tgt = possible[idx]
            inn = self._innovation_counter.get_innovation(src, tgt)
            net.connections[inn] = ConnectionGene(
                innovation_number=inn,
                source_id=src,
                target_id=tgt,
                weight=float(self.rng.uniform(-1.0, 1.0)),
                enabled=True,
            )

    # ------------------------------------------------------------------
    # Phenotype decoding
    # ------------------------------------------------------------------

    def get_phenotype(self) -> npt.NDArray[Any]:
        """Decode the CPPN into a ``(max_narms, 6)`` phenotype array.

        Applies repair to the decoded phenotype if enabled.
        """
        phenotype = decode_cppn_to_phenotype(
            self.genome,
            num_segments=self.num_segments,
            arm_limit=self.max_narms,
            parameter_limits=self.parameter_limits,
        )
        if self.repair_enabled:
            phenotype = self.repair_operator.repair(phenotype)
        return phenotype

    # ------------------------------------------------------------------
    # GenomeHandler interface
    # ------------------------------------------------------------------

    def generate_random_population(
        self, population_size: int
    ) -> List[CPPNNeatDroneGenomeHandler]:
        population: List[CPPNNeatDroneGenomeHandler] = []
        for _ in range(population_size):
            handler = CPPNNeatDroneGenomeHandler(
                genome=None,
                num_segments=self.num_segments,
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=self.parameter_limits,
                initial_hidden_nodes=self.initial_hidden_nodes,
                init_topology=self.init_topology,
                prob_add_node=self.prob_add_node,
                prob_add_connection=self.prob_add_connection,
                prob_remove_node=self.prob_remove_node,
                prob_remove_connection=self.prob_remove_connection,
                prob_mutate_weights=self.prob_mutate_weights,
                prob_mutate_activation=self.prob_mutate_activation,
                prob_toggle_connection=self.prob_toggle_connection,
                weight_perturb_std=self.weight_perturb_std,
                weight_replace_prob=self.weight_replace_prob,
                weight_range=self.weight_range,
                bias_perturb_std=self.bias_perturb_std,
                bias_replace_prob=self.bias_replace_prob,
                bias_range=self.bias_range,
                repair=self.repair_enabled,
                enable_collision_repair=self.enable_collision_repair,
                propeller_radius=self.propeller_radius,
                inner_boundary_radius=self.inner_boundary_radius,
                outer_boundary_radius=self.outer_boundary_radius,
                max_repair_iterations=self.max_repair_iterations,
                repair_step_size=self.repair_step_size,
                propeller_tolerance=self.propeller_tolerance,
                rng=self.rng,
            )
            population.append(handler)
        return population

    def crossover(self, other: CPPNNeatDroneGenomeHandler) -> CPPNNeatDroneGenomeHandler:
        """NEAT-style aligned crossover of two CPPN genomes."""
        child_net = crossover_cppn(
            self.genome, other.genome,
            self.fitness, other.fitness,
            self.rng,
        )
        return CPPNNeatDroneGenomeHandler(
            genome=child_net,
            num_segments=self.num_segments,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            initial_hidden_nodes=self.initial_hidden_nodes,
            init_topology=self.init_topology,
            prob_add_node=self.prob_add_node,
            prob_add_connection=self.prob_add_connection,
            prob_remove_node=self.prob_remove_node,
            prob_remove_connection=self.prob_remove_connection,
            prob_mutate_weights=self.prob_mutate_weights,
            prob_mutate_activation=self.prob_mutate_activation,
            prob_toggle_connection=self.prob_toggle_connection,
            weight_perturb_std=self.weight_perturb_std,
            weight_replace_prob=self.weight_replace_prob,
            weight_range=self.weight_range,
            bias_perturb_std=self.bias_perturb_std,
            bias_replace_prob=self.bias_replace_prob,
            bias_range=self.bias_range,
            repair=self.repair_enabled,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
            rng=self.rng,
        )

    def crossover_population(
        self,
        population1: List[GenomeHandler],
        population2: List[GenomeHandler],
    ) -> List[GenomeHandler]:
        """Perform NEAT crossover on paired populations."""
        assert len(population1) == len(population2)
        return [p1.crossover(p2) for p1, p2 in zip(population1, population2)]

    def mutate(self) -> None:
        """Mutate the CPPN genome in-place."""
        mutate_cppn(
            self.genome,
            self._innovation_counter,
            self.rng,
            prob_add_node=self.prob_add_node,
            prob_add_connection=self.prob_add_connection,
            prob_remove_node=self.prob_remove_node,
            prob_remove_connection=self.prob_remove_connection,
            prob_mutate_weights=self.prob_mutate_weights,
            prob_mutate_activation=self.prob_mutate_activation,
            prob_toggle_connection=self.prob_toggle_connection,
            weight_perturb_std=self.weight_perturb_std,
            weight_replace_prob=self.weight_replace_prob,
            weight_range=self.weight_range,
            bias_perturb_std=self.bias_perturb_std,
            bias_replace_prob=self.bias_replace_prob,
            bias_range=self.bias_range,
        )

    def copy(self) -> CPPNNeatDroneGenomeHandler:
        return CPPNNeatDroneGenomeHandler(
            genome=self.genome,
            num_segments=self.num_segments,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            initial_hidden_nodes=self.initial_hidden_nodes,
            init_topology=self.init_topology,
            prob_add_node=self.prob_add_node,
            prob_add_connection=self.prob_add_connection,
            prob_remove_node=self.prob_remove_node,
            prob_remove_connection=self.prob_remove_connection,
            prob_mutate_weights=self.prob_mutate_weights,
            prob_mutate_activation=self.prob_mutate_activation,
            prob_toggle_connection=self.prob_toggle_connection,
            weight_perturb_std=self.weight_perturb_std,
            weight_replace_prob=self.weight_replace_prob,
            weight_range=self.weight_range,
            bias_perturb_std=self.bias_perturb_std,
            bias_replace_prob=self.bias_replace_prob,
            bias_range=self.bias_range,
            repair=self.repair_enabled,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
            rng=self.rng,
        )

    def is_valid(self) -> bool:
        """Decode the CPPN to phenotype and check validity."""
        phenotype = self.get_phenotype()
        arm_count = int(np.sum(~np.isnan(phenotype[:, 0])))
        return self.min_narms <= arm_count <= self.max_narms

    def compatibility_distance(self, other: CPPNNeatDroneGenomeHandler) -> float:
        """NEAT compatibility distance based on CPPN topology and weights."""
        return cppn_compatibility_distance(self.genome, other.genome)

    def repair(self) -> None:
        """No-op — repair is applied to the decoded phenotype in get_phenotype()."""
        pass
