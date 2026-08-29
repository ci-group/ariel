"""Drone Blueprint — ARIEL-native intermediate representation for drones.

A typed tree (NetworkX DiGraph) carrying continuous SE(3) transforms and
physical parameters. Sits between any drone genotype and any phenotype
backend (MuJoCo propellers list, MJCF, USD/Isaac Lab, CAD/STL).

Pipeline:
    genome  --decoder-->  DroneBlueprint  --backend-->  phenotype
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx

from ariel.simulation.drone.propeller_data import get_propeller_specs


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


# ---------- arm cross sections ----------
#
# Each cross-section class describes the *shape* of an arm's cross-section
# (perpendicular to the arm's local +X axis). They expose a uniform
# interface used by ArmNode (mass / inertia derivation) and by backends
# (geometry emission, via isinstance dispatch).

@dataclass
class CylindricalCrossSection:
    """Solid circular cross-section."""
    type: str = "Cylindrical"    # discriminator for from_dict
    radius: float = 0.005        # m

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    def principal_inertia(self, length: float, mass: float) -> tuple[float, float, float]:
        # Solid cylinder along local +X. Ixx is the axial moment.
        Ixx = 0.5 * mass * self.radius ** 2
        Iyy = Izz = mass * (3 * self.radius ** 2 + length ** 2) / 12
        return (Ixx, Iyy, Izz)


@dataclass
class HollowTubeCrossSection:
    """Hollow circular cross-section (tube). Default = 8mm OD / 6mm ID,
    matching propeller_data.BEAM_DENSITY when density = 1500 kg/m³."""
    type: str = "HollowTube"
    outer_radius: float = 0.004  # m
    inner_radius: float = 0.003  # m

    @property
    def area(self) -> float:
        return math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)

    def principal_inertia(self, length: float, mass: float) -> tuple[float, float, float]:
        # Hollow cylinder along local +X.
        sum_sq = self.outer_radius ** 2 + self.inner_radius ** 2
        Ixx = 0.5 * mass * sum_sq
        Iyy = Izz = mass * (3 * sum_sq + length ** 2) / 12
        return (Ixx, Iyy, Izz)


@dataclass
class RectangularCrossSection:
    """Solid rectangular (box) cross-section."""
    type: str = "Rectangular"
    width: float = 0.01          # m (along arm's local +Y)
    thickness: float = 0.005     # m (along arm's local +Z)

    @property
    def area(self) -> float:
        return self.width * self.thickness

    def principal_inertia(self, length: float, mass: float) -> tuple[float, float, float]:
        # Box (length × width × thickness) along local +X.
        Ixx = mass * (self.width ** 2 + self.thickness ** 2) / 12
        Iyy = mass * (length ** 2 + self.thickness ** 2) / 12
        Izz = mass * (length ** 2 + self.width ** 2) / 12
        return (Ixx, Iyy, Izz)


CrossSection = Union[
    CylindricalCrossSection,
    HollowTubeCrossSection,
    RectangularCrossSection,
]


@dataclass
class ArmNode:
    type: str = "Arm"
    length: float = 0.18         # m
    density: float = 1500.0      # kg/m³ — carbon-fiber-ish
    cross_section: CrossSection = field(default_factory=HollowTubeCrossSection)
    pose: Pose = field(default_factory=Pose)   # attachment frame on parent (CorePlate)

    @property
    def mass(self) -> float:
        """Derived: density × cross-section area × length."""
        return self.density * self.cross_section.area * self.length

    @property
    def inertia_diag(self) -> tuple[float, float, float]:
        """Principal moments (Ixx, Iyy, Izz) about COM, arm along local +X."""
        return self.cross_section.principal_inertia(self.length, self.mass)


@dataclass
class MotorNode:
    type: str = "Motor"
    pose: Pose = field(default_factory=Pose)   # pose on parent Arm tip
    spin: str = "ccw"            # "cw" | "ccw"
    propsize: int = 5            # inches; PROPELLER_LIBRARY key

    @property
    def mass(self) -> float:
        """Lumped motor + propeller mass from PROPELLER_LIBRARY."""
        return get_propeller_specs(self.propsize)["mass"]

    @property
    def radius(self) -> float:
        """Visual / collision cylinder radius from PROPELLER_LIBRARY."""
        return get_propeller_specs(self.propsize)["motor_radius"]

    @property
    def thickness(self) -> float:
        """Visual / collision cylinder half-length from PROPELLER_LIBRARY."""
        return get_propeller_specs(self.propsize)["motor_thickness"]

    @property
    def inertia_diag(self) -> tuple[float, float, float]:
        """Solid-cylinder principal moments. Spin axis = local +Z."""
        m, r, h = self.mass, self.radius, 2 * self.thickness
        Ixx = Iyy = m * (3 * r ** 2 + h ** 2) / 12
        Izz = 0.5 * m * r ** 2
        return (Ixx, Iyy, Izz)


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
        cross_section_map = {
            "Cylindrical": CylindricalCrossSection,
            "HollowTube": HollowTubeCrossSection,
            "Rectangular": RectangularCrossSection,
        }
        for entry in d["nodes"]:
            data = dict(entry["data"])
            cls_ = type_map[data["type"]]
            # rebuild Pose sub-dataclass where present
            if "pose" in data and isinstance(data["pose"], dict):
                data["pose"] = Pose(**data["pose"])
            # rebuild cross_section sub-dataclass where present
            if "cross_section" in data and isinstance(data["cross_section"], dict):
                cs_data = dict(data["cross_section"])
                cs_cls = cross_section_map[cs_data["type"]]
                data["cross_section"] = cs_cls(**cs_data)
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
