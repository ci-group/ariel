"""Drone Blueprint — ARIEL-native intermediate representation for drones.

A typed tree (NetworkX DiGraph) carrying continuous SE(3) transforms and
physical parameters. Sits between any drone genotype and any phenotype
backend (MuJoCo propellers list, MJCF, USD/Isaac Lab, CAD/STL).

Pipeline:
    genome  --decoder-->  DroneBlueprint  --backend-->  phenotype
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx


# ---------- node payloads ----------

@dataclass
class Pose:
    """SE(3) pose relative to parent node. Translation in metres, rpy in radians."""
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CorePlateNode:
    type: str = "CorePlate"
    mass: float = 0.4
    shape: str = "disc"          # "disc" | "box"
    radius: float = 0.05         # disc radius (m)
    thickness: float = 0.01      # disc thickness (m)


@dataclass
class ArmNode:
    type: str = "Arm"
    length: float = 0.18         # m
    pose: Pose = field(default_factory=Pose)   # attachment frame on parent (CorePlate)


@dataclass
class MotorNode:
    type: str = "Motor"
    pose: Pose = field(default_factory=Pose)   # pose on parent Arm tip
    spin: str = "ccw"            # "cw" | "ccw"
    propsize: int = 5            # inches; looked up in propeller_data for kf/km


@dataclass
class RotorNode:
    type: str = "Rotor"
    radius: float = 0.0635       # m  (5" prop ≈ 0.127 m diameter)
    pitch: float = 0.045         # m
    blades: int = 2


@dataclass
class SensorNode:
    type: str = "Sensor"
    sensor_type: str = "imu"     # "imu" | "camera" | "range"
    pose: Pose = field(default_factory=Pose)


# ---------- blueprint container ----------

class DroneBlueprint:
    """Tree of typed drone-component nodes, JSON-serialisable.

    Internally a NetworkX DiGraph; root is the unique CorePlate node.
    Each node stores its dataclass payload under the `data` attribute.
    """

    def __init__(self) -> None:
        self.g: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0
        self.root_id: Optional[int] = None

    # ----- construction -----
    def add(self, payload: Any, parent: Optional[int] = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self.g.add_node(nid, data=payload)
        if parent is not None:
            self.g.add_edge(parent, nid)
        elif self.root_id is None:
            if not isinstance(payload, CorePlateNode):
                raise ValueError("First node added must be a CorePlateNode (root).")
            self.root_id = nid
        else:
            raise ValueError("Blueprint already has a root; pass parent= for further nodes.")
        return nid

    # ----- traversal helpers -----
    def nodes_of_type(self, type_name: str) -> list[int]:
        return [n for n, d in self.g.nodes(data=True) if d["data"].type == type_name]

    def children(self, n: int) -> list[int]:
        return list(self.g.successors(n))

    def payload(self, n: int) -> Any:
        return self.g.nodes[n]["data"]

    # ----- I/O -----
    def to_dict(self) -> dict:
        # NetworkX node-link with dataclass payloads serialised inline
        def encode(d: Any) -> dict:
            return asdict(d) if hasattr(d, "__dataclass_fields__") else d

        return {
            "root": self.root_id,
            "nodes": [
                {"id": n, "data": encode(d["data"])}
                for n, d in self.g.nodes(data=True)
            ],
            "edges": list(self.g.edges()),
        }

    def save_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "DroneBlueprint":
        bp = cls()
        bp.root_id = d["root"]
        type_map = {
            "CorePlate": CorePlateNode,
            "Arm": ArmNode,
            "Motor": MotorNode,
            "Rotor": RotorNode,
            "Sensor": SensorNode,
        }
        for entry in d["nodes"]:
            data = dict(entry["data"])
            cls_ = type_map[data["type"]]
            # rebuild Pose sub-dataclass where present
            if "pose" in data and isinstance(data["pose"], dict):
                data["pose"] = Pose(**data["pose"])
            payload = cls_(**data)
            bp.g.add_node(entry["id"], data=payload)
            bp._next_id = max(bp._next_id, entry["id"] + 1)
        bp.g.add_edges_from(d["edges"])
        return bp

    @classmethod
    def load_json(cls, path: Path | str) -> "DroneBlueprint":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ----- pretty -----
    def summary(self) -> str:
        lines = [f"DroneBlueprint ({self.g.number_of_nodes()} nodes)"]
        for n in nx.dfs_preorder_nodes(self.g, self.root_id):
            depth = int(nx.shortest_path_length(self.g, self.root_id, n))
            p = self.payload(n)
            lines.append("  " * depth + f"#{n} {p}")
        return "\n".join(lines)
