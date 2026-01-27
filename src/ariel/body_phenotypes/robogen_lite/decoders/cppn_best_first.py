import numpy as np
import numpy.typing as npt
from rich.console import Console
import networkx as nx
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
    NUM_OF_TYPES_OF_MODULES,
)
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
            ModuleFaces.FRONT: (1, 0, 0), ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0), ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1), ModuleFaces.LEFT: (0, 0, -1),
        }

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (parent_pos[0] + delta[0], parent_pos[1] + delta[1], parent_pos[2] + delta[2])

    def decode(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        module_data = {}
        
        core_id, core_pos, core_type, core_rot = IDX_OF_CORE, (0, 0, 0), ModuleType.CORE, ModuleRotationsIdx.DEG_0
        robot_graph.add_node(core_id, type=core_type.name, rotation=core_rot.name)
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {'pos': core_pos, 'type': core_type, 'rot': core_rot}
        
        # The frontier now contains ALL modules with potential open faces.
        frontier = [core_id]
        next_module_id = 1

        # Check for the max module count.
        while len(robot_graph) < self.max_modules:
            potential_connections = []
            
            # At each step, we check the ENTIRE frontier of all existing modules.
            for parent_id in frontier:
                parent_pos = module_data[parent_id]['pos']
                parent_type = module_data[parent_id]['type']
                for face in ModuleFaces:
                    
                    if face not in ALLOWED_FACES[parent_type]:
                        break
                    
                    child_pos = self._get_child_coords(parent_pos, face)
                    
                    if child_pos in occupied_coords:
                        break

                    cppn_inputs = list(parent_pos) + list(child_pos)
                    raw_outputs = self.cppn_genome.activate(cppn_inputs)
                    
                    conn_score = raw_outputs[0]
                    type_scores = np.array(raw_outputs[1:1+NUM_OF_TYPES_OF_MODULES])
                    rot_scores = np.array(raw_outputs[1+NUM_OF_TYPES_OF_MODULES:])

                    type_probs = softmax(type_scores)

                    type_probs[ModuleType.NONE.value] = -1.0 # Ignore NONE if that's the output
                    type_probs[ModuleType.CORE.value] = -1.0 # Ignore CORE if that's the output

                    child_type = ModuleType(np.argmax(type_probs))
                    child_rot = ModuleRotationsIdx(np.argmax(softmax(rot_scores)))

                    if face in ALLOWED_FACES[child_type] and child_rot in ALLOWED_ROTATIONS[child_type]:
                        potential_connections.append({
                            'score': conn_score, 'parent_id': parent_id, 'child_pos': child_pos,
                            'child_type': child_type, 'child_rot': child_rot, 'face': face,
                        })

            if not potential_connections:
                console.log("[yellow]Decoder stalled: No valid connections found anywhere on the robot.[/yellow]")
                break

            best_conn = max(potential_connections, key=lambda x: x['score'])
            
            child_id = next_module_id
            robot_graph.add_node(child_id, type=best_conn['child_type'].name, rotation=best_conn['child_rot'].name)
            robot_graph.add_edge(best_conn['parent_id'], child_id, face=best_conn['face'].name)
            
            occupied_coords[best_conn['child_pos']] = child_id
            module_data[child_id] = {'pos': best_conn['child_pos'], 'type': best_conn['child_type'], 'rot': best_conn['child_rot']}
            
            # I no longer remove the parent, I just add the new child.
            # (I think this makes snakes less likely)
            frontier.append(child_id)
            next_module_id += 1

        return robot_graph