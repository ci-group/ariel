from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import SubprocessHandler

__all__ = ['get_subprocess_handler']

def get_subprocess_handler(entrypoint: str, args: tuple, env: dict[str, str], stdout: str, stderr: str, local_rank_id: int) -> SubprocessHandler: ...
