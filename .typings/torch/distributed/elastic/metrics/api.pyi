import abc
from _typeshed import Incomplete
from typing import NamedTuple

__all__ = ['MetricsConfig', 'MetricHandler', 'ConsoleMetricHandler', 'NullMetricHandler', 'MetricStream', 'configure', 'getStream', 'prof', 'profile', 'put_metric', 'publish_metric', 'get_elapsed_time_ms', 'MetricData']

class MetricData(NamedTuple):
    timestamp: Incomplete
    group_name: Incomplete
    name: Incomplete
    value: Incomplete

class MetricsConfig:
    __slots__: Incomplete
    params: Incomplete
    def __init__(self, params: dict[str, str] | None = None) -> None: ...

class MetricHandler(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def emit(self, metric_data: MetricData): ...

class ConsoleMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData): ...

class NullMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData): ...

class MetricStream:
    group_name: Incomplete
    handler: Incomplete
    def __init__(self, group_name: str, handler: MetricHandler) -> None: ...
    def add_value(self, metric_name: str, metric_value: int): ...

def configure(handler: MetricHandler, group: str | None = None): ...
def getStream(group: str): ...
def prof(fn=None, group: str = 'torchelastic'):
    '''
    @profile decorator publishes duration.ms, count, success, failure metrics for the function that it decorates.

    The metric name defaults to the qualified name (``class_name.def_name``) of the function.
    If the function does not belong to a class, it uses the leaf module name instead.

    Usage

    ::

     @metrics.prof
     def x():
         pass


     @metrics.prof(group="agent")
     def y():
         pass
    '''
def profile(group=None):
    '''
    @profile decorator adds latency and success/failure metrics to any given function.

    Usage

    ::

     @metrics.profile("my_metric_group")
     def some_function(<arguments>):
    '''
def put_metric(metric_name: str, metric_value: int, metric_group: str = 'torchelastic'):
    '''
    Publish a metric data point.

    Usage

    ::

     put_metric("metric_name", 1)
     put_metric("metric_name", 1, "metric_group_name")
    '''
def publish_metric(metric_group: str, metric_name: str, metric_value: int): ...
def get_elapsed_time_ms(start_time_in_seconds: float):
    """Return the elapsed time in millis from the given start time."""

# Names in __all__ with no definition:
#   MetricData
