import torch
from _typeshed import Incomplete
from torch.utils import benchmark as benchmark
from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer as SpectralOpFuzzer
from typing import NamedTuple

def _dim_options(ndim): ...
def run_benchmark(name: str, function: object, dtype: torch.dtype, seed: int, device: str, samples: int, probability_regular: float): ...

class Benchmark(NamedTuple):
    name: Incomplete
    function: Incomplete
    dtype: Incomplete

BENCHMARKS: Incomplete
BENCHMARK_MAP: Incomplete
BENCHMARK_NAMES: Incomplete
DEVICE_NAMES: Incomplete

def _output_csv(file, results) -> None: ...
