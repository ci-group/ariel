import numpy as np
import typing as tp
from .bender import bender as bender
from .mode_converter import mode_converter as mode_converter

first_time_ceviche: bool

def gambas_function(x: np.ndarray, benchmark_type: int = 0, discretize: bool = False, wantgrad: bool = False, wantfields: bool = False) -> tp.Any:
    """
    x = 2d or 3d array of scalars
     Inputs:
    1. benchmark_type = 0 1 2 or 3, depending on which benchmark you want
    2. discretize = True if we want to force x to be in {0,1} (just checks sign(x-0.5) )
    3. wantgrad = True if we want to know the gradient
    4. wantfields = True if we want the fields of the simulation
     Returns:
    1. the loss (to be minimized)
    2. the gradient or none (depending on wantgrad)
    3. the fields or none (depending on wantfields
    """
