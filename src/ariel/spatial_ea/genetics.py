"""Genetic operators for the HyperNEAT genomes used by the spatial EA.

Operators come in two layers: genome-level functions that work on plain CPPN
genome dictionaries, and individual-level wrappers that also handle identity
and parentage bookkeeping. The spatial phase and the incubation phase share the
genome-level layer.

Notes
-----
    * Randomness goes through the ``numpy.random`` and ``random`` module-level
      generators rather than a private stream, so that seeding a worker process
      with ``np.random.seed`` / ``random.seed`` reproduces a run.
    * ``crossover_genomes`` merges the node sets of both parents. The research
      prototype's spatial-phase crossover kept only one parent's nodes while
      taking connections from both, which produced connections referencing
      nodes that did not exist; those connections were then silently dropped
      during evaluation. The prototype's own incubation phase already did the
      merge correctly, and that is the behaviour implemented here.

"""

# Standard library
import copy
import random
from typing import Any

# Third-party libraries
import numpy as np

# Local libraries
from ariel.spatial_ea.hyperneat import (
    ACTIVATION_FUNCTIONS,
    CPPNConnection,
    CPPNGenome,
    CPPNNode,
)
from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
WEIGHT_PERTURB_PROBABILITY = 0.9
WEIGHT_CLIP = 3.0
INITIAL_WEIGHT_SCALE = 3.0
REPLACEMENT_WEIGHT_SCALE = 2.0
HIDDEN_NODE_PROBABILITY = 0.5
INPUT_TO_HIDDEN_PROBABILITY = 0.7
DIRECT_CONNECTION_PROBABILITY = 0.4
EXTRA_CONNECTION_PROBABILITY = 0.3
MAX_EXTRA_CONNECTIONS_PER_INPUT = 2
CPPN_OUTPUT_ACTIVATIONS = (
    "sine",
    "tanh",
    "gaussian",
    "sigmoid",
    "linear",
    "relu",
    "abs",
)


# -- Genome construction -------------------------------------------------------
def create_initial_hyperneat_genome(
    num_inputs: int = 4,
    num_outputs: int = 1,
    activation: str = "sine",
) -> CPPNGenome:
    """Create a deliberately diverse initial CPPN genome.

    Initial populations use a wide weight distribution, a randomised output
    activation, probabilistic hidden nodes and extra random connections, so
    that the starting population spans a broad range of behaviours rather than
    clustering around a single minimal topology.

    Parameters
    ----------
    num_inputs
        Number of CPPN inputs, four for a pair of 2D substrate coordinates.
    num_outputs
        Number of CPPN outputs. Only the first output is read as a weight.
    activation
        Output activation. The sentinel ``"sine"`` requests a random choice
        from :data:`CPPN_OUTPUT_ACTIVATIONS`; any other value is used as-is.

    Returns
    -------
        A CPPN genome with ``nodes`` and ``connections`` entries.
    """
    del num_outputs  # Single-output CPPNs; kept for signature compatibility.

    nodes: list[CPPNNode] = [
        CPPNNode(node_id=i, activation="linear", layer=0)
        for i in range(num_inputs)
    ]
    connections: list[CPPNConnection] = []

    if activation == "sine":
        output_activation = str(np.random.choice(CPPN_OUTPUT_ACTIVATIONS))
    else:
        output_activation = activation

    if np.random.random() < HIDDEN_NODE_PROBABILITY:
        num_hidden = int(np.random.randint(1, 3))
        hidden_ids = [num_inputs + h for h in range(num_hidden)]
        nodes.extend(
            CPPNNode(
                node_id=hidden_id,
                activation=str(
                    np.random.choice(list(ACTIVATION_FUNCTIONS.keys())),
                ),
                layer=1,
            )
            for hidden_id in hidden_ids
        )

        output_id = num_inputs + num_hidden
        nodes.append(
            CPPNNode(node_id=output_id, activation=output_activation, layer=2),
        )

        for i in range(num_inputs):
            connections.extend(
                CPPNConnection(
                    from_node=i,
                    to_node=hidden_id,
                    weight=float(
                        np.random.randn() * INITIAL_WEIGHT_SCALE,
                    ),
                    enabled=True,
                )
                for hidden_id in hidden_ids
                if np.random.random() < INPUT_TO_HIDDEN_PROBABILITY
            )

        connections.extend(
            CPPNConnection(
                from_node=hidden_id,
                to_node=output_id,
                weight=float(np.random.randn() * INITIAL_WEIGHT_SCALE),
                enabled=True,
            )
            for hidden_id in hidden_ids
        )

        # Allow some connections to skip the hidden layer entirely.
        connections.extend(
            CPPNConnection(
                from_node=i,
                to_node=output_id,
                weight=float(np.random.randn() * INITIAL_WEIGHT_SCALE),
                enabled=True,
            )
            for i in range(num_inputs)
            if np.random.random() < DIRECT_CONNECTION_PROBABILITY
        )
    else:
        output_id = num_inputs
        nodes.append(
            CPPNNode(node_id=output_id, activation=output_activation, layer=1),
        )

        connections.extend(
            CPPNConnection(
                from_node=i,
                to_node=output_id,
                weight=float(np.random.randn() * INITIAL_WEIGHT_SCALE),
                enabled=True,
            )
            for i in range(num_inputs)
        )

        for i in range(num_inputs):
            outgoing = sum(1 for c in connections if c.from_node == i)
            if (
                np.random.random() < EXTRA_CONNECTION_PROBABILITY
                and outgoing < MAX_EXTRA_CONNECTIONS_PER_INPUT
            ):
                connections.append(
                    CPPNConnection(
                        from_node=i,
                        to_node=output_id,
                        weight=float(np.random.randn() * INITIAL_WEIGHT_SCALE),
                        enabled=True,
                    ),
                )

    return {"nodes": nodes, "connections": connections}


# -- Genome-level operators ----------------------------------------------------
def crossover_genomes(
    genotype1: CPPNGenome,
    genotype2: CPPNGenome,
) -> CPPNGenome:
    """Recombine two CPPN genomes, NEAT style.

    Connections are keyed by their ``(from_node, to_node)`` pair, which stands
    in for an innovation number. Matching connections are drawn at random from
    either parent; disjoint connections are inherited from whichever parent has
    them. Nodes are the union of both parents' node sets, so every inherited
    connection has both endpoints present.

    Parameters
    ----------
    genotype1
        First parent genome.
    genotype2
        Second parent genome.

    Returns
    -------
        A single offspring genome.
    """
    connections1 = {
        (c.from_node, c.to_node): c for c in genotype1["connections"]
    }
    connections2 = {
        (c.from_node, c.to_node): c for c in genotype2["connections"]
    }

    matching = set(connections1) & set(connections2)
    disjoint1 = set(connections1) - set(connections2)
    disjoint2 = set(connections2) - set(connections1)

    offspring_connections: list[CPPNConnection] = []
    for key in matching:
        chosen = (
            connections1[key] if random.random() < 0.5 else connections2[key]
        )
        offspring_connections.append(copy.deepcopy(chosen))
    offspring_connections.extend(
        copy.deepcopy(connections1[key]) for key in disjoint1
    )
    offspring_connections.extend(
        copy.deepcopy(connections2[key]) for key in disjoint2
    )

    nodes_by_id: dict[int, CPPNNode] = {}
    for node in genotype1["nodes"]:
        nodes_by_id[node.node_id] = copy.deepcopy(node)
    for node in genotype2["nodes"]:
        if node.node_id not in nodes_by_id:
            nodes_by_id[node.node_id] = copy.deepcopy(node)

    return {
        "nodes": list(nodes_by_id.values()),
        "connections": offspring_connections,
    }


def _mutate_weights(
    genome: CPPNGenome,
    weight_mutation_rate: float,
    weight_mutation_power: float,
    *,
    clip_weights: bool,
) -> None:
    """Perturb or replace connection weights in place.

    Parameters
    ----------
    genome
        Genome to mutate.
    weight_mutation_rate
        Probability that each individual weight is touched.
    weight_mutation_power
        Standard deviation of the Gaussian perturbation.
    clip_weights
        Whether to clamp perturbed weights to ``±WEIGHT_CLIP``.
    """
    for conn in genome["connections"]:
        if random.random() >= weight_mutation_rate:
            continue

        if random.random() < WEIGHT_PERTURB_PROBABILITY:
            conn.weight += float(np.random.normal(0, weight_mutation_power))
            if clip_weights:
                conn.weight = float(
                    np.clip(conn.weight, -WEIGHT_CLIP, WEIGHT_CLIP),
                )
        else:
            conn.weight = float(np.random.randn() * REPLACEMENT_WEIGHT_SCALE)


def _add_connection(genome: CPPNGenome) -> None:
    """Add one feed-forward connection between previously unlinked nodes.

    Parameters
    ----------
    genome
        Genome to mutate in place. Left unchanged when every valid
        feed-forward pair is already connected.
    """
    nodes: list[CPPNNode] = genome["nodes"]
    existing = {(c.from_node, c.to_node) for c in genome["connections"]}

    possible = [
        (source.node_id, target.node_id)
        for source in nodes
        for target in nodes
        if source.layer < target.layer
        and (source.node_id, target.node_id) not in existing
    ]
    if not possible:
        return

    from_id, to_id = random.choice(possible)
    genome["connections"].append(
        CPPNConnection(
            from_node=from_id,
            to_node=to_id,
            weight=float(np.random.normal(0, 1.0)),
            enabled=True,
        ),
    )


def _add_node(genome: CPPNGenome) -> None:
    """Split an enabled connection with a new hidden node.

    The new node is placed strictly between the endpoints of the connection it
    splits. When those endpoints sit in adjacent layers, every node at or above
    the insertion point is pushed one layer deeper to make room, which keeps
    the genome feed-forward and keeps the output layer the deepest one.

    Parameters
    ----------
    genome
        Genome to mutate in place. Left unchanged when no enabled connection
        exists or when a connection references a missing node.
    """
    enabled = [c for c in genome["connections"] if c.enabled]
    if not enabled:
        return

    conn_to_split = random.choice(enabled)
    nodes: list[CPPNNode] = genome["nodes"]
    source_node = next(
        (n for n in nodes if n.node_id == conn_to_split.from_node),
        None,
    )
    target_node = next(
        (n for n in nodes if n.node_id == conn_to_split.to_node),
        None,
    )
    if source_node is None or target_node is None:
        return

    conn_to_split.enabled = False

    new_layer = (source_node.layer + target_node.layer) // 2
    if new_layer == source_node.layer:
        new_layer = source_node.layer + 1
        for node in nodes:
            if node.layer >= new_layer and node.node_id != source_node.node_id:
                node.layer += 1

    new_node_id = max(n.node_id for n in nodes) + 1
    nodes.append(
        CPPNNode(
            node_id=new_node_id,
            activation=random.choice(list(ACTIVATION_FUNCTIONS.keys())),
            layer=new_layer,
        ),
    )

    genome["connections"].extend([
        CPPNConnection(
            from_node=source_node.node_id,
            to_node=new_node_id,
            weight=1.0,
            enabled=True,
        ),
        CPPNConnection(
            from_node=new_node_id,
            to_node=target_node.node_id,
            weight=conn_to_split.weight,
            enabled=True,
        ),
    ])


def mutate_genome(
    genome: CPPNGenome,
    weight_mutation_rate: float = 0.8,
    weight_mutation_power: float = 0.5,
    add_connection_rate: float = 0.05,
    add_node_rate: float = 0.03,
    *,
    clip_weights: bool = True,
) -> CPPNGenome:
    """Mutate a CPPN genome in place.

    Parameters
    ----------
    genome
        Genome to mutate.
    weight_mutation_rate
        Probability of mutating each connection weight.
    weight_mutation_power
        Standard deviation of the weight perturbation.
    add_connection_rate
        Probability of adding one new connection.
    add_node_rate
        Probability of splitting one connection with a new node.
    clip_weights
        Whether perturbed weights are clamped to ``±WEIGHT_CLIP``.

    Returns
    -------
        The same genome object, mutated.
    """
    _mutate_weights(
        genome,
        weight_mutation_rate,
        weight_mutation_power,
        clip_weights=clip_weights,
    )

    if random.random() < add_connection_rate:
        _add_connection(genome)

    if random.random() < add_node_rate:
        _add_node(genome)

    return genome


# -- Individual-level operators ------------------------------------------------
def clone_individual(
    individual: SpatialIndividual,
    next_unique_id: int,
    generation: int,
    initial_energy: float = 100.0,
) -> tuple[SpatialIndividual, int]:
    """Copy an individual's genome into a fresh identity.

    Parameters
    ----------
    individual
        Individual to clone.
    next_unique_id
        Identifier to assign to the clone.
    generation
        Generation the clone is born into.
    initial_energy
        Starting energy of the clone.

    Returns
    -------
    clone
        The new individual.
    next_unique_id
        The identifier counter, advanced past the clone.
    """
    clone = SpatialIndividual(
        unique_id=next_unique_id,
        generation=generation,
        genotype=copy.deepcopy(individual.genotype),
        energy=initial_energy,
        parent_ids=(
            [individual.unique_id] if individual.unique_id is not None else []
        ),
    )
    return clone, next_unique_id + 1


def crossover_hyperneat(
    parent1: SpatialIndividual,
    parent2: SpatialIndividual,
    next_unique_id: int,
    generation: int,
    initial_energy: float = 100.0,
) -> tuple[SpatialIndividual, SpatialIndividual, int]:
    """Produce two offspring from two parents.

    Parameters
    ----------
    parent1
        First parent.
    parent2
        Second parent.
    next_unique_id
        Identifier to assign to the first child.
    generation
        Generation the offspring are born into.
    initial_energy
        Starting energy of the offspring.

    Returns
    -------
    child1
        First offspring.
    child2
        Second offspring.
    next_unique_id
        The identifier counter, advanced past both children.
    """
    parent_ids = [
        parent_id
        for parent_id in (parent1.unique_id, parent2.unique_id)
        if parent_id is not None
    ]

    children: list[SpatialIndividual] = []
    for _ in range(2):
        children.append(
            SpatialIndividual(
                unique_id=next_unique_id,
                generation=generation,
                genotype=crossover_genomes(parent1.genotype, parent2.genotype),
                energy=initial_energy,
                parent_ids=list(parent_ids),
            ),
        )
        next_unique_id += 1

    return children[0], children[1], next_unique_id


def mutate_hyperneat(
    individual: SpatialIndividual,
    next_unique_id: int,
    weight_mutation_rate: float = 0.8,
    weight_mutation_power: float = 0.5,
    add_connection_rate: float = 0.05,
    add_node_rate: float = 0.03,
    initial_energy: float = 100.0,
) -> tuple[SpatialIndividual, int]:
    """Return a mutated copy of an individual under a fresh identity.

    Parameters
    ----------
    individual
        Individual to mutate.
    next_unique_id
        Identifier to assign to the mutant.
    weight_mutation_rate
        Probability of mutating each connection weight.
    weight_mutation_power
        Standard deviation of the weight perturbation.
    add_connection_rate
        Probability of adding one new connection.
    add_node_rate
        Probability of splitting one connection with a new node.
    initial_energy
        Starting energy of the mutant.

    Returns
    -------
    mutated
        The mutated individual.
    next_unique_id
        The identifier counter, advanced past the mutant.
    """
    mutated = SpatialIndividual(
        unique_id=next_unique_id,
        generation=individual.generation,
        genotype=mutate_genome(
            copy.deepcopy(individual.genotype),
            weight_mutation_rate,
            weight_mutation_power,
            add_connection_rate,
            add_node_rate,
        ),
        energy=initial_energy,
        parent_ids=(
            [individual.unique_id] if individual.unique_id is not None else []
        ),
    )
    return mutated, next_unique_id + 1


def genome_summary(genome: CPPNGenome) -> dict[str, Any]:
    """Summarise a genome's topology.

    Parameters
    ----------
    genome
        Genome to describe.

    Returns
    -------
        Node count, connection count and enabled-connection count.
    """
    connections: list[CPPNConnection] = genome.get("connections", [])
    return {
        "num_nodes": len(genome.get("nodes", [])),
        "num_connections": len(connections),
        "num_enabled": sum(1 for c in connections if c.enabled),
    }
