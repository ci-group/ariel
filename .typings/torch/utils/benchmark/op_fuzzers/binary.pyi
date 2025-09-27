from _typeshed import Incomplete
from torch.utils.benchmark import FuzzedParameter as FuzzedParameter, FuzzedTensor as FuzzedTensor, Fuzzer as Fuzzer, ParameterAlias as ParameterAlias

_MIN_DIM_SIZE: int
_MAX_DIM_SIZE: Incomplete
_POW_TWO_SIZES: Incomplete

class BinaryOpFuzzer(Fuzzer):
    def __init__(self, seed, dtype=..., cuda: bool = False) -> None: ...
