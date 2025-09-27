from .observer import ObserverBase, _PartialWrapper
from _typeshed import Incomplete
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from typing import NamedTuple

__all__ = ['QConfig', 'QConfigDynamic', 'default_qconfig', 'default_debug_qconfig', 'default_per_channel_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig', 'float16_static_qconfig', 'per_channel_dynamic_qconfig', 'float_qparams_weight_only_qconfig', 'float_qparams_weight_only_qconfig_4bit', 'default_quint8_weight_qconfig', 'default_qat_qconfig', 'default_dynamic_qat_qconfig', 'default_weight_only_qconfig', 'default_activation_only_qconfig', 'default_qat_qconfig_v2', 'default_reuse_input_qconfig', 'default_symmetric_qnnpack_qconfig', 'default_per_channel_symmetric_qnnpack_qconfig', 'default_symmetric_qnnpack_qat_qconfig', 'default_per_channel_symmetric_qnnpack_qat_qconfig', 'default_embedding_qat_qconfig', 'default_embedding_qat_qconfig_4bit', 'get_default_qconfig', 'get_default_qat_qconfig', 'get_default_qconfig_dict', 'get_default_qat_qconfig_dict', 'QConfigAny', 'qconfig_equals']

class QConfig(NamedTuple('QConfig', [('activation', Incomplete), ('weight', Incomplete)])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8),
      )

    """
    __slots__: Incomplete
    def __new__(cls, activation, weight): ...

class QConfigDynamic(NamedTuple('QConfigDynamic', [('activation', Incomplete), ('weight', Incomplete)])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """
    __slots__: Incomplete
    def __new__(cls, activation=..., weight=...): ...

default_qconfig: Incomplete
default_debug_qconfig: Incomplete
default_per_channel_qconfig: Incomplete
default_dynamic_qconfig: Incomplete
float16_dynamic_qconfig: Incomplete
float16_static_qconfig: Incomplete
per_channel_dynamic_qconfig: Incomplete
float_qparams_weight_only_qconfig: Incomplete
float_qparams_weight_only_qconfig_4bit: Incomplete
default_qat_qconfig: Incomplete
default_dynamic_qat_qconfig: Incomplete
default_weight_only_qconfig: Incomplete
default_activation_only_qconfig: Incomplete
default_qat_qconfig_v2: Incomplete
default_reuse_input_qconfig: Incomplete

def get_default_qconfig(backend: str = 'x86', version: int = 0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """

default_symmetric_qnnpack_qconfig: Incomplete
default_per_channel_symmetric_qnnpack_qconfig: Incomplete
default_embedding_qat_qconfig: Incomplete
default_embedding_qat_qconfig_4bit: Incomplete
default_quint8_weight_qconfig: Incomplete

def get_default_qat_qconfig(backend: str = 'x86', version: int = 1):
    """
    Returns the default QAT qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.
      * `version`: version, for backwards compatibility. Can be `None` or `1`.

    Return:
        qconfig
    """

default_symmetric_qnnpack_qat_qconfig: Incomplete
default_per_channel_symmetric_qnnpack_qat_qconfig: Incomplete

def get_default_qconfig_dict(backend: str = 'x86', version: int = 0): ...
def get_default_qat_qconfig_dict(backend: str = 'x86', version: int = 1): ...
QConfigAny = QConfig | None
_ObserverOrFakeQuantizeConstructor = _PartialWrapper | type[ObserverBase] | type[FakeQuantizeBase]

def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
