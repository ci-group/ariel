import networkx as nx
import numpy as np
import numpy.typing as npt
from collections import deque
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

class MorphologyDecoderCubePruning:
    """Decodes a CPPN by pre-computing a 3D score cube and pruning sequentially."""

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
        
        # The 'Cube': Maps (parent_pos, face) -> {score, child_pos, child_type, child_rot}
        self.score_cube = {}

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces) -> tuple:
        delta = self.face_deltas[face]
        return (
            parent_pos[0] + delta[0],
            parent_pos[1] + delta[1],
            parent_pos[2] + delta[2],
        )

    def _build_score_cube(self) -> None:
        """
        Eagerly evaluates the CPPN for all possible positions up to a 
        Manhattan distance of max_modules, generating the full 'cube' of scores.
        """
        visited_pos = set()
        # Queue stores: (current_coordinate, current_depth_from_core)
        queue = deque([((0, 0, 0), 0)])  
        total_queries = 0
        valid_entries = 0
        
        while queue:
            current_pos, depth = queue.popleft()
            
            if current_pos in visited_pos:
                continue
            visited_pos.add(current_pos)
            
            # Stop expanding the cube if we've reached the max possible snake-length
            if depth >= self.max_modules:
                continue
                
            for face in ModuleFaces:
                child_pos = self._get_child_coords(current_pos, face)
                
                # Query the CPPN for this specific 3D coordinate transition
                cppn_inputs = list(current_pos) + list(child_pos)
                raw_outputs = self.cppn_genome.activate(cppn_inputs)
                total_queries += 1

                conn_score = raw_outputs[0]
                type_scores = np.array(raw_outputs[1 : 1 + NUM_OF_TYPES_OF_MODULES])
                rot_scores = np.array(raw_outputs[1 + NUM_OF_TYPES_OF_MODULES :])

                type_probs = softmax(type_scores)
                type_probs[ModuleType.CORE.value] = -1.0  # Core can only be at origin
                
                child_type = ModuleType(np.argmax(type_probs))
                
                # Select rotation only from those allowed for this module type
                allowed_rots = ALLOWED_ROTATIONS[child_type]
                allowed_rot_indices = [r.value for r in allowed_rots]
                
                # Mask out disallowed rotations before selecting
                rot_scores_masked = np.array(rot_scores, dtype=float)
                for i in range(len(rot_scores_masked)):
                    if i not in allowed_rot_indices:
                        rot_scores_masked[i] = -np.inf
                
                child_rot = ModuleRotationsIdx(np.argmax(rot_scores_masked))

                # Save to the cube (all scores, filtering happens during construction)
                self.score_cube[(current_pos, face)] = {
                    "score": conn_score,
                    "child_pos": child_pos,
                    "child_type": child_type,
                    "child_rot": child_rot
                }
                valid_entries += 1
                
                if child_pos not in visited_pos:
                    queue.append((child_pos, depth + 1))
        
        console.log(f"[cyan]Cube built: {total_queries} CPPN queries, {valid_entries} entries stored[/cyan]")

    def decode(self) -> nx.DiGraph:
        """Constructs the robot by pruning paths from the pre-computed cube."""
        
        # 1. Generate the universe of possible scores
        self._build_score_cube()
        
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        
        core_id, core_pos, core_type, core_rot = (
            IDX_OF_CORE,
            (0, 0, 0),
            ModuleType.CORE,
            ModuleRotationsIdx.DEG_0,
        )
        robot_graph.add_node(core_id, type=core_type.name, rotation=core_rot.name)
        occupied_coords[core_pos] = core_id
        
        # 2. Setup the available attachments pool with Core's allowed faces
        available_attachments = []
        for face in ALLOWED_FACES[core_type]:
            cube_data = self.score_cube.get((core_pos, face))
            if cube_data:
                available_attachments.append({
                    "parent_id": core_id,
                    "parent_pos": core_pos,
                    "face": face,
                    **cube_data
                })
        
        console.log(f"[cyan]Initial attachments from core: {len(available_attachments)} candidates[/cyan]")
        if not available_attachments:
            console.log("[yellow]No valid candidates from core; decoder will return core-only.[/yellow]")
                
        next_module_id = 1
        rejections = {"spatial": 0, "rotation": 0}
        
        # 3. Best-first construction with sequential pruning
        while len(robot_graph) < self.max_modules and available_attachments:
            
            # Sort to pop the highest CPPN connection score
            available_attachments.sort(key=lambda x: x["score"], reverse=True)
            best_conn = available_attachments.pop(0)
            
            child_pos = best_conn["child_pos"]
            child_type = best_conn["child_type"]
            child_rot = best_conn["child_rot"]
            face = best_conn["face"]
            
            # Prune: Spatial intersection (position already occupied)
            if child_pos in occupied_coords:
                rejections["spatial"] += 1
                continue

            # Valid Attachment Found: Add to Graph
            child_id = next_module_id
            robot_graph.add_node(child_id, type=child_type.name, rotation=child_rot.name)
            robot_graph.add_edge(best_conn["parent_id"], child_id, face=face.name)
            occupied_coords[child_pos] = child_id
            
            # ADD NEW ATTACHMENTS (Sequential Pruning)
            # We ONLY look up the cube for faces allowed by this specific newly-placed module.
            # If a hinge is placed, this loop naturally prunes all faces except FRONT.
            for next_face in ALLOWED_FACES[child_type]:
                cube_data = self.score_cube.get((child_pos, next_face))
                
                if cube_data and cube_data["child_pos"] not in occupied_coords:
                    available_attachments.append({
                        "parent_id": child_id,
                        "parent_pos": child_pos,
                        "face": next_face,
                        **cube_data
                    })
                    
            next_module_id += 1

        console.log(f"[cyan]Decoder: {len(robot_graph)} modules, {rejections['spatial']} spatial rejections, {rejections['rotation']} rotation rejections[/cyan]")
        if len(robot_graph) == 1:
            console.log("[yellow]Decoder stalled: No valid connections found from core.[/yellow]")

        return robot_graph