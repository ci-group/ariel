from _typeshed import Incomplete
from typing import NamedTuple

TORCH_AVAILABLE: bool

class SystemEnv(NamedTuple):
    torch_version: Incomplete
    is_debug_build: Incomplete
    cuda_compiled_version: Incomplete
    gcc_version: Incomplete
    clang_version: Incomplete
    cmake_version: Incomplete
    os: Incomplete
    libc_version: Incomplete
    python_version: Incomplete
    python_platform: Incomplete
    is_cuda_available: Incomplete
    cuda_runtime_version: Incomplete
    cuda_module_loading: Incomplete
    nvidia_driver_version: Incomplete
    nvidia_gpu_models: Incomplete
    cudnn_version: Incomplete
    pip_version: Incomplete
    pip_packages: Incomplete
    conda_packages: Incomplete
    hip_compiled_version: Incomplete
    hip_runtime_version: Incomplete
    miopen_runtime_version: Incomplete
    caching_allocator_config: Incomplete
    is_xnnpack_available: Incomplete
    cpu_info: Incomplete

COMMON_PATTERNS: Incomplete
NVIDIA_PATTERNS: Incomplete
CONDA_PATTERNS: Incomplete
PIP_PATTERNS: Incomplete

def run(command):
    """Return (return-code, stdout, stderr)."""
def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
def run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
def get_conda_packages(run_lambda, patterns=None): ...
def get_gcc_version(run_lambda): ...
def get_clang_version(run_lambda): ...
def get_cmake_version(run_lambda): ...
def get_nvidia_driver_version(run_lambda): ...
def get_gpu_info(run_lambda): ...
def get_running_cuda_version(run_lambda): ...
def get_cudnn_version(run_lambda):
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
def get_nvidia_smi(): ...
def get_cpu_info(run_lambda): ...
def get_platform(): ...
def get_mac_version(run_lambda): ...
def get_windows_version(run_lambda): ...
def get_lsb_version(run_lambda): ...
def check_release_file(run_lambda): ...
def get_os(run_lambda): ...
def get_python_platform(): ...
def get_libc_version(): ...
def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
def get_cachingallocator_config(): ...
def get_cuda_module_loading_config(): ...
def is_xnnpack_available(): ...
def get_env_info():
    """
    Collects environment information to aid in debugging.

    The returned environment information contains details on torch version, is debug build
    or not, cuda compiled version, gcc version, clang version, cmake version, operating
    system, libc version, python version, python platform, CUDA availability, CUDA
    runtime version, CUDA module loading config, GPU model and configuration, Nvidia
    driver version, cuDNN version, pip version and versions of relevant pip and
    conda packages, HIP runtime version, MIOpen runtime version,
    Caching allocator config, XNNPACK availability and CPU information.

    Returns:
        SystemEnv (namedtuple): A tuple containining various environment details
            and system information.
    """

env_info_fmt: Incomplete

def pretty_str(envinfo): ...
def get_pretty_env_info():
    """
    Returns a pretty string of environment information.

    This function retrieves environment information by calling the `get_env_info` function
    and then formats the information into a human-readable string. The retrieved environment
    information is listed in the document of `get_env_info`.
    This function is used in `python collect_env.py` that should be executed when reporting a bug.

    Returns:
        str: A pretty string of the environment information.
    """
def main() -> None: ...
