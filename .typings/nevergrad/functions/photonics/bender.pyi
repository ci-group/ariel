from .functions import addupml2d as addupml2d, block as block, yeeder2d as yeeder2d
from _typeshed import Incomplete

c0: int
epsi0: float
mu0: float
Z0: Incomplete
micrometers: int
nanometers: Incomplete

def bender(X, ev_out=..., ev_in=...):
    """
    Computes the conversion efficiency between the mode of index ev_in
    in the input wg, into the mode of index ev_out in the output wg,
    given a central block of dimensions:
        centerWG_w = 2 * micrometers        # Width
        centerWG_L = 2 * micrometers          # Length
    described in X float array of size (centerWG_w x dx , centerWG_L x dy)
    X is the permittivities
    Default ev values compute fundamental mode to first order mode
    """
