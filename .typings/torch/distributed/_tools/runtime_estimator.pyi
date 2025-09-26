import torch
from _typeshed import Incomplete
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any, Callable
from typing_extensions import Self

__all__ = ['RuntimeEstimator']

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS

class RuntimeEstimator(TorchDispatchMode):
    '''
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it aggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (Dict[str, Dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by \'fw\' and \'bw\'.
        mod_fw_pre_order (List[str]): List of module FQNs in pre-forward execution order.
        mod_bw_pre_order (List[str]): List of module FQNs in pre-backward execution order.
        mod_fw_post_order (List[str]): List of module FQNs in post-forward execution order.
        mod_bw_post_order (List[str]): List of module FQNs in post-backward execution order.
        total_runtime (float): The total estimated runtime in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) Currently wrapper tensor sub-classes such as ``DTensor`` won\'t produce correct estimates. We plan to support
            them in future PRs.
        3) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
            support this in future PRs.

    Example usage:

        .. code-block:: python

            runtime_estimator = RuntimeEstimator()
            with FakeTensorMode():
                module = ...
                optimizer = ...
                inp = ...
                with runtime_estimator(estimate_mode_type="operator-level-cost-model"):
                    loss = module(inp)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                runtime_estimator.display_modulewise_stats()
    '''
    _float_types: set[torch.dtype]
    _no_fallback_kernel: set[torch._ops._OpNamespace]
    fake_mode: FakeTensorMode
    _estimate: Callable
    _estimate_mode_type: str
    _mod_tracker: Incomplete
    mod_runtimes: dict[str, dict[str, float]]
    mod_fw_pre_order: list[str]
    mod_bw_pre_order: list[str]
    mod_fw_post_order: list[str]
    mod_bw_post_order: list[str]
    total_runtime: float
    def __init__(self) -> None: ...
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(cls, func, args, kwargs, orig_not_implemented_exception):
        """
        Runs and benchmarks a fallback kernel for a given function.

        Args:
            func (Callable): The function to benchmark.
            args (Tuple): The arguments to pass to the function.
            kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
            orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
                is not implemented.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs) -> tuple[Any, float]:
        """
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
    @classmethod
    def _roofline_estimate(cls, func, args, kwargs) -> tuple[Any, float]:
        """
        Estimates the runtime of a function using a roofline cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
    def display_modulewise_stats(self, depth: int = 2) -> None:
        """
        Displays module-wise statistics collected by ``RuntimeEstimator``.

        Prints the pre-forward and pre-backward execution orders.
        Displays the module-wise forward and backward runtimes in milliseconds.

        Args:
            depth (int): The maximum depth of module hierarchy to display (default to 2).
        """
    def __torch_dispatch__(self, func, types, args=..., kwargs=None): ...
    def __call__(self, estimate_mode_type: str) -> Self:
        '''
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            RuntimeEstimator: The runtime estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        '''
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any) -> None: ...
