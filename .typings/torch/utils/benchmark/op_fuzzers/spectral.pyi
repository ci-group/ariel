from torch.utils import benchmark

__all__ = ['SpectralOpFuzzer']

class SpectralOpFuzzer(benchmark.Fuzzer):
    def __init__(self, *, seed: int, dtype=..., cuda: bool = False, probability_regular: float = 1.0) -> None: ...
