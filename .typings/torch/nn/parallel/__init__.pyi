from torch.nn.parallel.data_parallel import DataParallel as DataParallel, data_parallel as data_parallel
from torch.nn.parallel.distributed import DistributedDataParallel as DistributedDataParallel
from torch.nn.parallel.parallel_apply import parallel_apply as parallel_apply
from torch.nn.parallel.replicate import replicate as replicate
from torch.nn.parallel.scatter_gather import gather as gather, scatter as scatter

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel', 'DataParallel', 'DistributedDataParallel']

class DistributedDataParallelCPU(DistributedDataParallel): ...
