import torch
from typing import TypeVar

T = TypeVar('T')

def all_same_mode(modes): ...
no_dispatch = torch._C._DisableTorchDispatch
