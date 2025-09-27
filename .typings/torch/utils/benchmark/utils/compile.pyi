import torch
from typing import Any, Callable

__all__ = ['bench_all', 'benchmark_compile']

def benchmark_compile(model: torch.nn.Module | Callable, sample_input: torch.Tensor | Any, num_iters: int = 5, backend: str | None = None, mode: str | None = 'default', optimizer: torch.optim.Optimizer | None = None, loss_fn: torch.nn.Module | Callable | None = None):
    """
        Use this utility to benchmark torch.compile
        """
def bench_all(model: torch.nn.Module | Callable, sample_input: torch.Tensor | Any, num_iters: int = 5, optimizer: torch.optim.Optimizer | None = None, loss_fn: torch.nn.Module | Callable | None = None):
    """
        This is a simple utility that can be used to benchmark torch.compile
        In particular it ensures that your GPU is setup to use tensor cores if it supports its
        It also tries out all the main backends and prints a table of results so you can easily compare them all
        Many of the backendds have their own optional dependencies so please pip install them seperately

        You will get one table for inference and another for training
        If you'd like to leverage this utility for training make sure to pass in a torch.optim.Optimizer

        The important warnings are
        Your GPU supports tensor cores
        we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`

        If a compilation fails for any reason including the dependency not being included
        then we will print Failed to compile {backend} with mode {mode}
        """
