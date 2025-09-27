import functools
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch as get_cuda_arch

@functools.cache
def gen_cutlass_presets() -> dict[int, dict[str, list[str]]]:
    """
    Generate cutlass presets for the given CUDA arch.
    """
