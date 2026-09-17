"""TreeGenome: minimal tree-based genome representation and conversion.

This module provides a simple JSON-serializable tree genome that converts
to a NetworkX DiGraph compatible with `construct_mjspec_from_graph`.
"""

from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List

import networkx as nx


@dataclass
class TreeGenome:
    # nodes: mapping id -> dict with keys: type (Enum.name), rotation (Enum.name),
    # and optional length (float)
    nodes: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # edges: list of {"parent": int, "child": int, "face": ModuleFaces.name}
    edges: List[Dict[str, typing.Any]] = field(default_factory=list)

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()

        for nid, nobj in self.nodes.items():
            # Add all node attributes to preserve optional morphology parameters
            g.add_node(
                nid,
                **nobj,
            )

        for e in self.edges:
            g.add_edge(
                e["parent"],
                e["child"],
                face=e["face"],
            )

        return g

    def to_dict(self) -> Dict[str, typing.Any]:
        return {
            "nodes": {
                str(k): v
                for k, v in self.nodes.items()
            },
            "edges": self.edges,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, typing.Any],
    ) -> "TreeGenome":
        nodes_raw = data.get("nodes", {})

        # Accept both string-keyed mapping and list
        nodes: Dict[int, Dict[str, Any]] = {}

        if isinstance(nodes_raw, dict):
            for k, v in nodes_raw.items():
                nodes[int(k)] = dict(v)

        elif isinstance(nodes_raw, list):
            for entry in nodes_raw:
                nid = int(entry["id"])

                # Preserve all node attributes except the identifier
                nodes[nid] = {
                    k: v
                    for k, v in entry.items()
                    if k != "id"
                }

        edges = data.get("edges", [])

        return cls(
            nodes=nodes,
            edges=edges,
        )

    def save_json(
        self,
        path: str,
    ) -> None:
        with open(path, "w") as fh:
            json.dump(
                self.to_dict(),
                fh,
                indent=2,
            )

    @classmethod
    def load_json(
        cls,
        path: str,
    ) -> "TreeGenome":
        with open(path, "r") as fh:
            data = json.load(fh)

        return cls.from_dict(data)