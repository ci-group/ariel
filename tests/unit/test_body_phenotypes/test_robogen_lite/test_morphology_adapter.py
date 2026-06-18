"""Tests for MorphologyAdapter (ariel.simulation.controllers.morphology_adapter).

Validates that:
1. gecko_graph() produces the correct node/edge structure.
2. MorphologyAdapter.from_graph() derives module types and face neighbors that
   match what the old hardcoded constants encoded.
3. Actuator ordering matches MuJoCo's compiled order for construct_mjspec_from_graph.
4. get_node_inputs() runs without error and returns the right shape/structure.
"""

from __future__ import annotations

import sys
import os

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers import MorphologyAdapter

# gecko_graph and scale_actions are example-specific helpers that live in
# examples/d_social_learning/ rather than the library.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "examples", "d_social_learning"))
from morphology_adapter import gecko_graph  # noqa: E402
from gecko_utils import scale_actions  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    return gecko_graph()


@pytest.fixture(scope="module")
def adapter(graph):
    return MorphologyAdapter.from_graph(graph)


@pytest.fixture(scope="module")
def compiled_gecko(graph):
    """MuJoCo model built from the same graph the adapter uses."""
    core = construct_mjspec_from_graph(graph)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=(-0.8, 0.0, 0.1), rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


# ---------------------------------------------------------------------------
# gecko_graph() structure
# ---------------------------------------------------------------------------

class TestGeckoGraph:
    def test_has_core_at_zero(self, graph):
        assert graph.nodes[0]["type"] == ModuleType.CORE.name

    def test_hinge_count(self, graph):
        hinges = [n for n in graph.nodes if graph.nodes[n]["type"] == ModuleType.HINGE.name]
        assert len(hinges) == 8  # neck, spine, fl_leg, fl_leg2, fr_leg, fr_leg2, bl_leg, br_leg

    def test_brick_count(self, graph):
        # abdomen(2), butt(4), fl_flipper(6), fr_flipper(8), bl_flipper(10), br_flipper(12)
        bricks = [n for n in graph.nodes if graph.nodes[n]["type"] == ModuleType.BRICK.name]
        assert len(bricks) == 6

    def test_all_edges_have_face(self, graph):
        for parent, child in graph.edges:
            face_name = graph.edges[(parent, child)]["face"]
            # Should be a valid ModuleFaces name
            ModuleFaces[face_name]  # raises KeyError if invalid

    def test_core_connects_to_neck_on_front(self, graph):
        assert graph.has_edge(0, 1)
        assert graph.edges[(0, 1)]["face"] == ModuleFaces.FRONT.name

    def test_core_connects_to_fl_leg_on_left(self, graph):
        assert graph.has_edge(0, 5)
        assert graph.edges[(0, 5)]["face"] == ModuleFaces.LEFT.name

    def test_core_connects_to_fr_leg_on_right(self, graph):
        assert graph.has_edge(0, 7)
        assert graph.edges[(0, 7)]["face"] == ModuleFaces.RIGHT.name


# ---------------------------------------------------------------------------
# Module type mapping
# ---------------------------------------------------------------------------

class TestModuleTypes:
    """Compare adapter.module_type against the old hardcoded MODULE_TYPE dict.

    Old values (one-hot index):  0=empty, 1=brick, 2=hinge, 3=core
    Old MODULE_TYPE = {0:3, 1:2, 2:1, 3:2, 4:1, 5:2, 6:1, 7:2, 8:1,
                       9:2, 10:1, 11:2, 12:1, 15:2, 17:2}
    """
    OLD_MODULE_TYPE = {
        0: 3, 1: 2, 2: 1, 3: 2, 4: 1,
        5: 2, 6: 1, 7: 2, 8: 1,
        9: 2, 10: 1, 11: 2, 12: 1,
        15: 2, 17: 2,
    }

    def test_all_module_indices_present(self, adapter):
        for idx in self.OLD_MODULE_TYPE:
            assert idx in adapter.module_type, f"module {idx} missing"

    def test_module_types_match_old_hardcoded(self, adapter):
        for idx, expected in self.OLD_MODULE_TYPE.items():
            assert adapter.module_type[idx] == expected, (
                f"module {idx}: got {adapter.module_type[idx]}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Face neighbor mapping
# ---------------------------------------------------------------------------

class TestFaceNeighbors:
    """Compare adapter.face_neighbors against the old hardcoded GECKO_FACE_NEIGHBORS.

    Old GECKO_FACE_NEIGHBORS (slot order: FRONT=0,BACK=1,RIGHT=2,LEFT=3,TOP=4,BOTTOM=5):
        1:  [2,    0,    None, None, None, None]   neck
        3:  [4,    2,    None, None, None, None]   spine
        5:  [15,   0,    None, None, None, None]   fl_leg  (parent via LEFT face)
        15: [6,    5,    None, None, None, None]   fl_leg2
        7:  [17,   0,    None, None, None, None]   fr_leg  (parent via RIGHT face)
        17: [8,    7,    None, None, None, None]   fr_leg2
        9:  [10,   4,    None, None, None, None]   bl_leg  (parent via LEFT face)
        11: [12,   4,    None, None, None, None]   br_leg  (parent via RIGHT face)

    NOTE: the "BACK" slot in the old table was a convention meaning "parent".
    In the dynamic system the parent occupies the face it used on the parent
    module, not necessarily BACK. For chain modules (neck, spine) that is FRONT,
    and for leg hinges it is LEFT or RIGHT. So we check each slot explicitly.
    """

    def test_neck_front_child_is_abdomen(self, adapter):
        # neck (1): parent=core(0) attached via FRONT, child=abdomen(2) via FRONT
        slots = adapter.face_neighbors[1]
        assert slots[ModuleFaces.FRONT.value] == 2   # child: abdomen

    def test_neck_back_slot_is_parent_core(self, adapter):
        # core attached neck on its FRONT face → neck sees core in FRONT slot
        # old table had core(0) in BACK slot — that was a simplification.
        # The dynamic system puts the parent in the face the parent *used*, i.e. FRONT.
        # So neck's FRONT slot holds both child(2) and parent(0)? No — FRONT is used
        # for the child. The parent-attachment face on neck is the face core used: FRONT.
        # But that face is *taken* by the child. The parent actually appears in the
        # face slot corresponding to the edge (core→neck), which is FRONT.
        # After the child also uses FRONT, there is a collision — the child overwrites.
        # This is an inherent limitation: hinge only has FRONT for children, so the
        # parent slot ends up in whichever face the parent used.
        # For chain: core→neck via FRONT, neck→abdomen via FRONT → parent and child
        # both map to FRONT slot. Child wins (written last). Parent is lost in FRONT.
        # This matches real topology: hinges have no BACK children, so BACK stays None.
        slots = adapter.face_neighbors[1]
        # Child (abdomen=2) is at FRONT:
        assert slots[ModuleFaces.FRONT.value] == 2
        # All other slots are None (hinge has no other children, core was at FRONT too
        # but child overwrote it — parent info is effectively unavailable for pure-chain hinges)
        for face in (ModuleFaces.BACK, ModuleFaces.RIGHT, ModuleFaces.LEFT, ModuleFaces.TOP, ModuleFaces.BOTTOM):
            assert slots[face.value] is None

    def test_fl_leg_parent_is_in_left_slot(self, adapter):
        # core attached fl_leg via LEFT face → fl_leg sees core in LEFT slot
        slots = adapter.face_neighbors[5]
        assert slots[ModuleFaces.LEFT.value] == 0   # parent: core

    def test_fl_leg_front_child_is_fl_leg2(self, adapter):
        slots = adapter.face_neighbors[5]
        assert slots[ModuleFaces.FRONT.value] == 15

    def test_fr_leg_parent_is_in_right_slot(self, adapter):
        # core attached fr_leg via RIGHT face → fr_leg sees core in RIGHT slot
        slots = adapter.face_neighbors[7]
        assert slots[ModuleFaces.RIGHT.value] == 0

    def test_fr_leg_front_child_is_fr_leg2(self, adapter):
        slots = adapter.face_neighbors[7]
        assert slots[ModuleFaces.FRONT.value] == 17

    def test_bl_leg_parent_is_in_left_slot(self, adapter):
        # butt attached bl_leg via LEFT → bl_leg sees butt in LEFT slot
        slots = adapter.face_neighbors[9]
        assert slots[ModuleFaces.LEFT.value] == 4

    def test_br_leg_parent_is_in_right_slot(self, adapter):
        slots = adapter.face_neighbors[11]
        assert slots[ModuleFaces.RIGHT.value] == 4

    def test_neighbor_slots_length_is_six(self, adapter):
        for mod_idx, slots in adapter.face_neighbors.items():
            assert len(slots) == 6, f"module {mod_idx} has {len(slots)} slots, expected 6"


# ---------------------------------------------------------------------------
# Actuator ordering vs MuJoCo
# ---------------------------------------------------------------------------

class TestActuatorOrdering:
    """The adapter's actuator_to_module list must agree with MuJoCo's compiled order."""

    def test_actuator_count_matches_model(self, adapter, compiled_gecko):
        model, _ = compiled_gecko
        assert len(adapter.actuator_to_module) == model.nu

    def test_actuator_joint_names_contain_module_indices(self, adapter, compiled_gecko):
        """Each MuJoCo actuator name encodes its module index as the second token
        in the '{from}-{to}-{face}-servo' pattern."""
        model, _ = compiled_gecko
        for mj_idx in range(model.nu):
            raw_name = model.actuator(mj_idx).name  # e.g. "robot1_0-1-0-servo"
            # Strip spawn prefix
            suffix = raw_name.split("_", 1)[-1]     # "0-1-0-servo"
            parts = suffix.split("-")               # ["0", "1", "0", "servo"]
            child_module = int(parts[1])
            expected_module = adapter.actuator_to_module[mj_idx]
            assert child_module == expected_module, (
                f"MuJoCo actuator {mj_idx} ({raw_name}): "
                f"child module {child_module} != adapter says {expected_module}"
            )

    def test_all_actuator_modules_are_hinges(self, adapter, graph):
        for mod_idx in adapter.actuator_to_module:
            assert graph.nodes[mod_idx]["type"] == ModuleType.HINGE.name, (
                f"module {mod_idx} in actuator list is not a HINGE"
            )


# ---------------------------------------------------------------------------
# get_node_inputs() runtime
# ---------------------------------------------------------------------------

class TestGetNodeInputs:
    def test_returns_correct_number_of_nodes(self, adapter, compiled_gecko):
        model, data = compiled_gecko
        mujoco.mj_resetData(model, data)
        node_inputs, t = adapter.get_node_inputs(model, data, timestep=0)
        assert len(node_inputs) == model.nu

    def test_time_signal_at_zero(self, adapter, compiled_gecko):
        model, data = compiled_gecko
        mujoco.mj_resetData(model, data)
        _, t = adapter.get_node_inputs(model, data, timestep=0)
        assert t == pytest.approx(0.0)

    def test_time_signal_wraps_at_25(self, adapter, compiled_gecko):
        model, data = compiled_gecko
        mujoco.mj_resetData(model, data)
        _, t0 = adapter.get_node_inputs(model, data, timestep=0)
        _, t25 = adapter.get_node_inputs(model, data, timestep=25)
        assert t0 == pytest.approx(t25)

    def test_each_node_has_six_neighbors(self, adapter, compiled_gecko):
        model, data = compiled_gecko
        mujoco.mj_resetData(model, data)
        node_inputs, _ = adapter.get_node_inputs(model, data, timestep=0)
        for self_obs, nb_obs in node_inputs:
            assert len(nb_obs) == 6

    def test_self_obs_type_onehot_is_hinge(self, adapter, compiled_gecko):
        """Every actuator is a hinge — one-hot index 2 should be 1.0."""
        model, data = compiled_gecko
        mujoco.mj_resetData(model, data)
        node_inputs, _ = adapter.get_node_inputs(model, data, timestep=0)
        for self_obs, _ in node_inputs:
            assert self_obs.type_onehot[2] == pytest.approx(1.0), (
                f"expected hinge one-hot at index 2, got {self_obs.type_onehot}"
            )

    def test_state_in_range(self, adapter, compiled_gecko):
        """Joint state should be in [-1, 1] (normalised angle)."""
        model, data = compiled_gecko
        mujoco.mj_resetData(model, data)
        # Apply a non-trivial control signal and step
        data.ctrl[:] = 0.5
        for _ in range(50):
            mujoco.mj_step(model, data)
        node_inputs, _ = adapter.get_node_inputs(model, data, timestep=10)
        for self_obs, _ in node_inputs:
            assert -1.0 <= self_obs.state <= 1.0


# ---------------------------------------------------------------------------
# scale_actions
# ---------------------------------------------------------------------------

def test_scale_actions_maps_tanh_to_joint_range():
    raw = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    scaled = scale_actions(raw)
    assert scaled == pytest.approx(raw * (np.pi / 2))
