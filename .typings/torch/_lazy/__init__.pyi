from .closure import add_step_closure as add_step_closure, run_step_closures as run_step_closures
from torch.utils._pytree import tree_flatten as tree_flatten, tree_unflatten as tree_unflatten

def mark_step(device: str = '', wait: bool = False):
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
def wait_device_ops(devices=None) -> None:
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
def sync_multi(tensors, devices) -> None:
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
def get_tensor_id(tensor):
    """Return a unique id of the lazy tensor maintained by LTC"""
def to_cpu(tensors, devices=None): ...
def save(tensors, *args, **kwargs) -> None: ...
