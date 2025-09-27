from collections.abc import Generator
from contextlib import contextmanager
from torch.distributed.elastic.multiprocessing.errors import record

__all__ = ['worker_main']

@record
@contextmanager
def worker_main() -> Generator[None, None, None]:
    '''
    This is a context manager that wraps your main entry function. This combines
    the existing ``errors.record`` logic as well as a new ``_WorkerServer`` that
    exposes handlers via a unix socket specified by
    ``Torch_WORKER_SERVER_SOCKET``.

    Example

    ::

     @worker_main()
     def main():
         pass


     if __name__ == "__main__":
         main()

    '''
