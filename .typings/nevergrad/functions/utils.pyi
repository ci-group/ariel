import numpy as np
from _typeshed import Incomplete

class Transform:
    """Defines a unique random transformation (index selection, translation, and optionally rotation)
    which can be applied to a point
    """
    indices: Incomplete
    translation: np.ndarray
    rotation_matrix: np.ndarray | None
    def __init__(self, indices: list[int], translation_factor: float = 1, rotation: bool = False, random_state: np.random.RandomState | None = None, expo: float = 1.0) -> None: ...
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
