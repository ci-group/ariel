import typing as tp
from . import base as base, optimizerlib as optimizerlib
from _typeshed import Incomplete
from nevergrad.common.tools import pytorch_import_fix as pytorch_import_fix

Optim = type[base.Optimizer] | base.ConfiguredOptimizer

class TorchOptimizer:
    '''Experimental helper to perform optimization using torch
    workflow with a nevergrad optimizer

    Parameters
    ----------
    parameters: iterable
        module parameters which need to be optimized
    cls: Optimizer-like object
        name of a nevergrad optimizer, or nevergrad optimizer class, or ConfiguredOptimizer instance
    bound: float
        values are clipped to [-bound, bound]

    Notes
    -----
    - This is experimental, the API may evolve
    - This does not support parallelization (multiple asks).


    Example
    -------
    ..code::python

        module = ...
        optimizer = helpers.TorchOptimizer(module.parameters(), "OnePlusOne")
        for x, y in batcher():
            loss = compute_loss(module(x), y)
            optimizer.step(loss)

    '''
    parameters: Incomplete
    optimizer: Incomplete
    candidate: Incomplete
    def __init__(self, parameters: tp.Iterable[tp.Any], cls: str | Optim, bound: float = 20.0) -> None: ...
    def _set_candidate(self) -> None: ...
    def step(self, loss: float) -> None: ...
