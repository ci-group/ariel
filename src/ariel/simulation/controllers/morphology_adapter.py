"""Graph-derived observation builder for ``DistributedMLP``.

``MorphologyAdapter`` is built from a robogen-lite NetworkX DiGraph and derives
actuator ordering, module types, and face-neighbour topology entirely from the
graph — no hardcoded constants.  Works for any valid robogen-lite morphology,
including evolved ones produced by the decoder pipeline.

Usage::

    from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
    from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
    from ariel.simulation.controllers.distributed_mlp import DistributedMLP

    adapter = MorphologyAdapter.from_graph(graph)   # derive topology once
    brain   = DistributedMLP(n_neighbors=6)
    # ...per control step:
    node_inputs, t = adapter.get_node_inputs(model, data, timestep)
    actions = brain.forward_all(node_inputs, t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.simulation.controllers.distributed_mlp import EMPTY_NODE, NodeObservation

if TYPE_CHECKING:
    from networkx import DiGraph

_K = len(ModuleFaces)  # 6 face slots per node

# Map ModuleType → one-hot index used in NodeObservation.type_onehot (size 5):
#   0=empty/none, 1=brick, 2=hinge, 3=core, 4=unused
_TYPE_TO_ONEHOT_IDX: dict[ModuleType, int] = {
    ModuleType.NONE:  0,
    ModuleType.BRICK: 1,
    ModuleType.HINGE: 2,
    ModuleType.CORE:  3,
}
_ONEHOT_SIZE = 5


def _onehot(type_idx: int) -> np.ndarray:
    oh = np.zeros(_ONEHOT_SIZE, dtype=np.float32)
    if 0 <= type_idx < _ONEHOT_SIZE:
        oh[type_idx] = 1.0
    return oh


@dataclass
class MorphologyAdapter:
    """Topology adapter for ``DistributedMLP`` observations.

    Derives actuator ordering, module types, and face-neighbour slots from a
    robogen-lite DiGraph so that ``get_node_inputs`` can produce the per-node
    observations ``DistributedMLP.forward_all`` expects, for any morphology.

    Build with ``MorphologyAdapter.from_graph(graph)`` rather than directly.

    Parameters
    ----------
    actuator_to_module:
        Module index for each MuJoCo actuator, in actuator index order.
        Length equals ``model.nu``.
    module_type:
        Mapping from module index to one-hot index
        (0=empty, 1=brick, 2=hinge, 3=core).
    face_neighbors:
        Mapping from module index to a 6-element list
        ``[FRONT, BACK, RIGHT, LEFT, TOP, BOTTOM]``.
        Each slot is a neighbour module index or ``None`` (→ ``EMPTY_NODE``).
    """

    actuator_to_module: list[int]
    module_type: dict[int, int]
    face_neighbors: dict[int, list[int | None]]
    _joint_name_to_module: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def from_graph(cls, graph: DiGraph) -> MorphologyAdapter:
        """Derive topology entirely from a robogen-lite DiGraph.

        The graph must follow the convention produced by
        ``construct_mjspec_from_graph`` / the decoder pipeline:
        nodes have integer keys with ``type`` (ModuleType name) and
        ``rotation`` attributes; edges are ``(parent, child)`` with a
        ``face`` (ModuleFaces name) attribute.

        Actuator ordering mirrors the edge-iteration order of
        ``construct_mjspec_from_graph``, which matches MuJoCo's actuator order.

        Parameters
        ----------
        graph:
            Robogen-lite morphology graph.

        Returns
        -------
        MorphologyAdapter
            Adapter ready to call ``get_node_inputs``.
        """
        module_type: dict[int, int] = {
            node: _TYPE_TO_ONEHOT_IDX[ModuleType[graph.nodes[node]["type"]]]
            for node in graph.nodes
        }

        face_neighbors: dict[int, list[int | None]] = {}
        for node in graph.nodes:
            slots: list[int | None] = [None] * _K
            for parent in graph.predecessors(node):
                face_idx = ModuleFaces[graph.edges[(parent, node)]["face"]].value
                slots[face_idx] = parent
            for child in graph.successors(node):
                face_idx = ModuleFaces[graph.edges[(node, child)]["face"]].value
                slots[face_idx] = child
            face_neighbors[node] = slots

        actuator_to_module: list[int] = [
            child
            for _parent, child in graph.edges
            if ModuleType[graph.nodes[child]["type"]] == ModuleType.HINGE
        ]

        # construct_mjspec_from_graph names joints "{from}-{to}-{face_val}-servo"
        joint_name_to_module: dict[str, int] = {
            f"{parent}-{child}-{ModuleFaces[graph.edges[(parent, child)]['face']].value}-servo": child
            for parent, child in graph.edges
            if ModuleType[graph.nodes[child]["type"]] == ModuleType.HINGE
        }

        return cls(
            actuator_to_module=actuator_to_module,
            module_type=module_type,
            face_neighbors=face_neighbors,
            _joint_name_to_module=joint_name_to_module,
        )

    def _read_joint_states(
        self, model: mj.MjModel, data: mj.MjData
    ) -> tuple[dict[int, float], dict[int, float]]:
        joint_pos: dict[int, float] = {}
        joint_vel: dict[int, float] = {}
        for i in range(model.njnt):
            jname = model.joint(i).name
            matched = next(
                (k for k in self._joint_name_to_module if jname == k or jname.endswith("_" + k)),
                None,
            )
            if matched is not None:
                mod_idx = self._joint_name_to_module[matched]
                joint_pos[mod_idx] = float(data.qpos[model.joint(i).qposadr[0]])
                joint_vel[mod_idx] = float(data.qvel[model.joint(i).dofadr[0]])
        return joint_pos, joint_vel

    def _make_node_obs(
        self,
        module_idx: int,
        joint_pos: dict[int, float],
        joint_vel: dict[int, float],
    ) -> NodeObservation:
        oh_idx = self.module_type.get(module_idx, 0)
        if oh_idx == 0:
            return EMPTY_NODE
        angle = joint_pos.get(module_idx, 0.0)
        vel = joint_vel.get(module_idx, 0.0)
        state = float(np.clip(angle / (np.pi / 2), -1.0, 1.0))
        return NodeObservation(
            vel_x=vel,
            vel_y=0.0,
            state=state,
            type_onehot=_onehot(oh_idx),
        )

    def get_node_inputs(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        timestep: int,
    ) -> tuple[list[tuple[NodeObservation, list[NodeObservation]]], float]:
        """Build ``(node_inputs, time_signal)`` for ``DistributedMLP.forward_all``.

        Parameters
        ----------
        model:
            Compiled MuJoCo model.
        data:
            Current MuJoCo data.
        timestep:
            Control step counter (not simulation step).  Used to derive a
            cyclic time signal in ``[0, 1]`` over a 25-step window.

        Returns
        -------
        tuple
            ``node_inputs`` is a list of ``(self_obs, neighbour_obs_list)``
            pairs, one per actuator.  ``time_signal`` is a float in ``[0, 1]``.
        """
        joint_pos, joint_vel = self._read_joint_states(model, data)
        time_signal = np.sin(2 * np.pi * timestep / 25)

        node_inputs: list[tuple[NodeObservation, list[NodeObservation]]] = []
        for mod_idx in self.actuator_to_module:
            self_obs = self._make_node_obs(mod_idx, joint_pos, joint_vel)
            nb_slots = self.face_neighbors.get(mod_idx, [None] * _K)
            nb_obs = [
                self._make_node_obs(nb, joint_pos, joint_vel) if nb is not None else EMPTY_NODE
                for nb in nb_slots
            ]
            node_inputs.append((self_obs, nb_obs))

        return node_inputs, time_signal
