import torch
from _typeshed import Incomplete
from torch import Tensor

__all__ = ['FloatFunctional', 'FXFloatFunctional', 'QFunctional']

class FloatFunctional(torch.nn.Module):
    """State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """
    activation_post_process: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x) -> None: ...
    def add(self, x: Tensor, y: Tensor) -> Tensor: ...
    def add_scalar(self, x: Tensor, y: float) -> Tensor: ...
    def mul(self, x: Tensor, y: Tensor) -> Tensor: ...
    def mul_scalar(self, x: Tensor, y: float) -> Tensor: ...
    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor: ...
    def add_relu(self, x: Tensor, y: Tensor) -> Tensor: ...
    def matmul(self, x: Tensor, y: Tensor) -> Tensor: ...

class FXFloatFunctional(torch.nn.Module):
    """module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """
    def forward(self, x) -> None: ...
    def add(self, x: Tensor, y: Tensor) -> Tensor: ...
    def add_scalar(self, x: Tensor, y: float) -> Tensor: ...
    def mul(self, x: Tensor, y: Tensor) -> Tensor: ...
    def mul_scalar(self, x: Tensor, y: float) -> Tensor: ...
    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor: ...
    def add_relu(self, x: Tensor, y: Tensor) -> Tensor: ...
    def matmul(self, x: Tensor, y: Tensor) -> Tensor: ...

class QFunctional(torch.nn.Module):
    """Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """
    scale: float
    zero_point: int
    activation_post_process: Incomplete
    def __init__(self) -> None: ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    def _get_name(self): ...
    def extra_repr(self): ...
    def forward(self, x) -> None: ...
    def add(self, x: Tensor, y: Tensor) -> Tensor: ...
    def add_scalar(self, x: Tensor, y: float) -> Tensor: ...
    def mul(self, x: Tensor, y: Tensor) -> Tensor: ...
    def mul_scalar(self, x: Tensor, y: float) -> Tensor: ...
    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor: ...
    def add_relu(self, x: Tensor, y: Tensor) -> Tensor: ...
    def matmul(self, x: Tensor, y: Tensor) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
