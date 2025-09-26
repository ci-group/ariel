import nevergrad as ng
import numpy as np
import typing as tp
from . import gambas as gambas, photonics as photonics
from .. import base as base
from _typeshed import Incomplete
from nevergrad.ops import mutations as mutations

ceviche = photonics.ceviche
gambas = gambas.gambas_function

def _make_parametrization(name: str, dimension: int, bounding_method: str = 'bouncing', rolling: bool = False, as_tuple: bool = False) -> ng.p.Parameter:
    '''Creates appropriate parametrization for a Photonics problem

    Parameters
    name: str
        problem name, among bragg, chirped, cf_photosic_realistic, cf_photosic_reference and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    bounding_method: str
        transform type for the bounding ("arctan", "tanh", "bouncing" or "clipping"see `Array.bounded`)
    as_tuple: bool
        whether we should use a Tuple of Array instead of a 2D-array.

    Returns
    -------
    Instrumentation
        the parametrization for the problem
    '''

class Photonics(base.ExperimentFunction):
    '''Function calling photonics code

    Parameters
    ----------
    name: str
        problem name, among bragg, chirped, cf_photosic_realistic, cf_photosic_reference and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    transform: str
        transform type for the bounding ("arctan", "tanh", "bouncing" or "clipping", see `Array.bounded`)

    Returns
    -------
    float
        the fitness

    Notes
    -----
    - You will require an Octave installation (with conda: "conda install -c conda-forge octave" then re-source dfconda.sh)
    - Each function requires from around 1 to 5 seconds to compute
    - OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 are enforced when spawning Octave because parallelization leads to
      deadlock issues here.

    Credit
    ------
    This module is based on code and ideas from:
    - Mamadou Aliou Barry
    - Marie-Claire Cambourieux
    - Rémi Pollès
    - Antoine Moreau
    from University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal.

    Publications
    ------------
    - Aliou Barry, Mamadou; Berthier, Vincent; Wilts, Bodo D.; Cambourieux, Marie-Claire; Pollès, Rémi;
      Teytaud, Olivier; Centeno, Emmanuel; Biais, Nicolas; Moreau, Antoine (2018)
      Evolutionary algorithms converge towards evolved biological photonic structures,
      https://arxiv.org/abs/1808.04689
    - Defrance, J., Lemaître, C., Ajib, R., Benedicto, J., Mallet, E., Pollès, R., Plumey, J.-P.,
      Mihailovic, M., Centeno, E., Ciracì, C., Smith, D.R. and Moreau, A. (2016)
      Moosh: A Numerical Swiss Army Knife for the Optics of Multilayers in Octave/Matlab. Journal of Open Research Software, 4(1), p.e13.
    '''
    name: Incomplete
    _as_tuple: Incomplete
    _base_func: tp.Callable[[np.ndarray], float]
    def __init__(self, name: str, dimension: int, bounding_method: str = 'clipping', rolling: bool = False, as_tuple: bool = False) -> None: ...
    def to_array(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray: ...
    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float: ...
    def _compute(self, *args: tp.Any, **kwargs: tp.Any) -> float: ...
