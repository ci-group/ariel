from torch.utils.benchmark import Timer as Timer
from torch.utils.benchmark.op_fuzzers.binary import BinaryOpFuzzer as BinaryOpFuzzer
from torch.utils.benchmark.op_fuzzers.unary import UnaryOpFuzzer as UnaryOpFuzzer

_MEASURE_TIME: float

def assert_dicts_equal(dict_0, dict_1) -> None:
    '''Builtin dict comparison will not compare numpy arrays.
    e.g.
        x = {"a": np.ones((2, 1))}
        x == x  # Raises ValueError
    '''
def run(n, stmt, fuzzer_cls) -> None: ...
def main() -> None: ...
