import dataclasses
import numpy as np
import pathlib

@dataclasses.dataclass(frozen=True)
class Msh:
    """MuJoCo legacy binary msh file."""
    vertex_positions: np.ndarray
    vertex_normals: np.ndarray
    vertex_texcoords: np.ndarray
    face_vertex_indices: np.ndarray
    @staticmethod
    def create(file: pathlib.Path) -> Msh:
        """Create a Msh object from a .msh file."""

def msh_to_obj(msh_file: pathlib.Path) -> str:
    """Convert a legacy .msh file to the .obj format."""
