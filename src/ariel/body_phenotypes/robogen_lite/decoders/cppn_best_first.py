import heapq

import networkx as nx
import numpy as np
import numpy.typing as npt
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_TYPES_OF_MODULES,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome

console = Console()


def softmax(raw_scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    e_x = np.exp(raw_scores - np.max(raw_scores))
    return e_x / e_x.sum()


class MorphologyDecoderBestFirst:
    """Decodes a CPPN using a true greedy, best-first search strategy."""

    def __init__(self, cppn_genome: Genome, max_modules: int = 20):
        self.cppn_genome = cppn_genome
        self.max_modules = max_modules
        self.face_deltas = {
            ModuleFaces.FRONT: (1, 0, 0),
            ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0),
            ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1),
            ModuleFaces.LEFT: (0, 0, -1),
        }

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (
            parent_pos[0] + delta[0],
            parent_pos[1] + delta[1],
            parent_pos[2] + delta[2],
        )

    def decode(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords: dict[tuple, int] = {}
        module_data: dict[int, dict] = {}

        core_id = IDX_OF_CORE
        core_pos = (0, 0, 0)
        core_type = ModuleType.CORE
        core_rot = ModuleRotationsIdx.DEG_0

        robot_graph.add_node(core_id, type=core_type.name, rotation=core_rot.name)
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {"pos": core_pos, "type": core_type, "rot": core_rot}

        # Max-heap (negate score); counter breaks ties for heap stability
        heap: list[tuple] = []
        _counter = 0

        def _enqueue_faces(parent_id: int) -> None:
            nonlocal _counter
            parent_pos = module_data[parent_id]["pos"]
            parent_type = module_data[parent_id]["type"]
            for face in ModuleFaces:
                if face not in ALLOWED_FACES[parent_type]:
                    continue
                child_pos = self._get_child_coords(parent_pos, face)
                if child_pos in occupied_coords:
                    continue

                cppn_inputs = list(parent_pos) + list(child_pos)
                raw_outputs = self.cppn_genome.activate(cppn_inputs)

                conn_score = raw_outputs[0]
                type_scores = np.array(raw_outputs[1: 1 + NUM_OF_TYPES_OF_MODULES])
                rot_scores = np.array(raw_outputs[1 + NUM_OF_TYPES_OF_MODULES:])

                type_probs = softmax(type_scores)
                type_probs[ModuleType.NONE.value] = -1.0
                type_probs[ModuleType.CORE.value] = -1.0
                child_type = ModuleType(np.argmax(type_probs))
                child_rot = ModuleRotationsIdx(np.argmax(softmax(rot_scores)))

                if (
                    face in ALLOWED_FACES[child_type]
                    and child_rot in ALLOWED_ROTATIONS[child_type]
                ):
                    heapq.heappush(heap, (
                        -conn_score,
                        _counter,
                        {
                            "parent_id": parent_id,
                            "child_pos": child_pos,
                            "child_type": child_type,
                            "child_rot": child_rot,
                            "face": face,
                        },
                    ))
                    _counter += 1

        _enqueue_faces(core_id)
        next_module_id = 1

        while len(robot_graph) < self.max_modules:
            # Pop until we find a connection whose target is still unoccupied
            best_conn = None
            while heap:
                _, _, conn = heapq.heappop(heap)
                if conn["child_pos"] not in occupied_coords:
                    best_conn = conn
                    break

            if best_conn is None:
                console.log(
                    "[yellow]Decoder stalled: No valid connections found anywhere on the robot.[/yellow]"
                )
                break

            child_id = next_module_id
            robot_graph.add_node(
                child_id,
                type=best_conn["child_type"].name,
                rotation=best_conn["child_rot"].name,
            )
            robot_graph.add_edge(
                best_conn["parent_id"], child_id, face=best_conn["face"].name
            )
            occupied_coords[best_conn["child_pos"]] = child_id
            module_data[child_id] = {
                "pos": best_conn["child_pos"],
                "type": best_conn["child_type"],
                "rot": best_conn["child_rot"],
            }

            # Only the new module has new open faces; existing frontier is already in the heap
            _enqueue_faces(child_id)
            next_module_id += 1

        return robot_graph
