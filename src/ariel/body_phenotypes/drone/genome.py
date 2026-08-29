"""Serialization between drone genomes and ARIEL's JSONIterable.

Two formats are supported:

1. SphericalNeatGenome (direct encoding) — fixed-size arm matrix + innovation
   IDs. Empty slots are JSON null (avoids NaN-in-JSON) preserving slot
   positions needed for NEAT crossover alignment::

       {
           "arms": [[mag, theta, phi, motor_theta, motor_phi, dir], ..., null, ...],
           "innovation_ids": [0, 1, -1, ...]
       }

2. CPPNNetwork (indirect encoding) — flat dict of nodes and connections.
   Enums are stored by ``.value``::

       {
           "nodes": [{"node_id": int, "node_type": str, "activation": str,
                      "bias": float, "output_index": int|None,
                      "input_label": str|None}, ...],
           "connections": [{"innovation_number": int, "source_id": int,
                            "target_id": int, "weight": float,
                            "enabled": bool}, ...],
           "next_node_id": int
       }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ariel.ec.drone.genome_handlers.cppn.network import CPPNNetwork
    from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
        SphericalNeatGenome,
    )


def serialize_genome(genome: "SphericalNeatGenome") -> dict[str, Any]:
    """Convert a SphericalNeatGenome to a JSON-serialisable dict.

    NaN arm slots become None (JSON null). Innovation IDs are stored
    inline alongside the arm array so crossover alignment survives the
    DB round-trip.
    """
    arms_list: list[list[float] | None] = []
    for i in range(genome.arms.shape[0]):
        if np.isnan(genome.arms[i, 0]):
            arms_list.append(None)
        else:
            arms_list.append(genome.arms[i].tolist())
    return {
        "arms": arms_list,
        "innovation_ids": genome.innovation_ids.tolist(),
    }


def deserialize_genome(data: dict[str, Any]) -> "SphericalNeatGenome":
    """Reconstruct a SphericalNeatGenome from a stored dict.

    None slots become NaN-padded arm rows. Innovation IDs are restored
    to their original integer array so NEAT crossover can align genes.
    """
    from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
        SphericalNeatGenome,
    )

    arms_list = data["arms"]
    max_narms = len(arms_list)
    arms = np.full((max_narms, 6), np.nan, dtype=np.float64)
    for i, arm in enumerate(arms_list):
        if arm is not None:
            arms[i] = arm
    innovation_ids = np.array(data["innovation_ids"], dtype=int)
    return SphericalNeatGenome(arms=arms, innovation_ids=innovation_ids)


def serialize_cppn_genome(genome: "CPPNNetwork") -> dict[str, Any]:
    """Convert a CPPNNetwork to a JSON-serialisable dict.

    Enum-valued fields (``node_type``, ``activation``) are stored by their
    ``.value`` string so the dict can be written straight into a SQLite JSON
    column without custom encoders.
    """
    nodes_out: list[dict[str, Any]] = []
    for node in genome.nodes.values():
        nodes_out.append(
            {
                "node_id": int(node.node_id),
                "node_type": node.node_type.value,
                "activation": node.activation.value,
                "bias": float(node.bias),
                "output_index": (
                    int(node.output_index) if node.output_index is not None else None
                ),
                "input_label": node.input_label,
            },
        )

    conns_out: list[dict[str, Any]] = []
    for conn in genome.connections.values():
        conns_out.append(
            {
                "innovation_number": int(conn.innovation_number),
                "source_id": int(conn.source_id),
                "target_id": int(conn.target_id),
                "weight": float(conn.weight),
                "enabled": bool(conn.enabled),
            },
        )

    return {
        "nodes": nodes_out,
        "connections": conns_out,
        "next_node_id": int(genome.next_node_id),
    }


def deserialize_cppn_genome(data: dict[str, Any]) -> "CPPNNetwork":
    """Reconstruct a CPPNNetwork from a stored dict."""
    from ariel.ec.drone.genome_handlers.cppn.network import (
        ActivationFunction,
        ConnectionGene,
        CPPNNetwork,
        NodeGene,
        NodeType,
    )

    net = CPPNNetwork()
    for entry in data["nodes"]:
        nid = int(entry["node_id"])
        net.nodes[nid] = NodeGene(
            node_id=nid,
            node_type=NodeType(entry["node_type"]),
            activation=ActivationFunction(entry["activation"]),
            bias=float(entry["bias"]),
            output_index=(
                int(entry["output_index"])
                if entry.get("output_index") is not None
                else None
            ),
            input_label=entry.get("input_label"),
        )

    for entry in data["connections"]:
        inn = int(entry["innovation_number"])
        net.connections[inn] = ConnectionGene(
            innovation_number=inn,
            source_id=int(entry["source_id"]),
            target_id=int(entry["target_id"]),
            weight=float(entry["weight"]),
            enabled=bool(entry["enabled"]),
        )

    net.next_node_id = int(data.get("next_node_id", 0))
    return net
