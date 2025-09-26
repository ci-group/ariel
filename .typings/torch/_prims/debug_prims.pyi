import contextlib
from collections.abc import Generator
from torch.utils._content_store import ContentStoreReader as ContentStoreReader

LOAD_TENSOR_READER: ContentStoreReader | None

@contextlib.contextmanager
def load_tensor_reader(loc) -> Generator[None]: ...
def register_debug_prims(): ...
