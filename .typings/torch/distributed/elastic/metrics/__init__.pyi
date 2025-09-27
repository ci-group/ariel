from torch.distributed.elastic.metrics.static_init import *
from .api import ConsoleMetricHandler as ConsoleMetricHandler, MetricData as MetricData, MetricHandler as MetricHandler, MetricsConfig as MetricsConfig, NullMetricHandler as NullMetricHandler, configure as configure, getStream as getStream, get_elapsed_time_ms as get_elapsed_time_ms, prof as prof, profile as profile, publish_metric as publish_metric, put_metric as put_metric

def initialize_metrics(cfg: MetricsConfig | None = None): ...
