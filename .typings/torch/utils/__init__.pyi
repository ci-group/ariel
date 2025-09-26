from _typeshed import Incomplete
from torch.utils.backend_registration import generate_methods_for_privateuse1_backend as generate_methods_for_privateuse1_backend, rename_privateuse1_backend as rename_privateuse1_backend
from torch.utils.cpp_backtrace import get_cpp_backtrace as get_cpp_backtrace
from torch.utils.throughput_benchmark import ThroughputBenchmark as ThroughputBenchmark

def set_module(obj, mod) -> None:
    """
    Set the module attribute on a python object for a given object for nicer printing
    """

cmake_prefix_path: Incomplete

def swap_tensors(t1, t2) -> None:
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
