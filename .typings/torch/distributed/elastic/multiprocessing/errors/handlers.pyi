from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler

__all__ = ['get_error_handler']

def get_error_handler() -> ErrorHandler: ...
