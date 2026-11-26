import mujoco

def load(filepath: str) -> mujoco.MjModel:
    """Loads and compiles a model file: XML, URDF, MJB, or MJZ/ZIP."""
def parse(filepath: str) -> mujoco.MjData:
    """Loads a model file and returns a fresh ``MjData`` for it."""
def dependencies(filepath: str) -> list[str]:
    """Returns the files a model file is built from, including ``filepath``.

  For XML/URDF: the file, its ``<include>`` and ``<model>`` trees and their
  mesh, skin, hfield and texture assets. Other formats: just the file.

  Args:
    filepath: The model file.
  """
