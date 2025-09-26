from _typeshed import Incomplete

class _BaseDatasetFetcher:
    dataset: Incomplete
    auto_collation: Incomplete
    collate_fn: Incomplete
    drop_last: Incomplete
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None: ...
    def fetch(self, possibly_batched_index) -> None: ...

class _IterableDatasetFetcher(_BaseDatasetFetcher):
    dataset_iter: Incomplete
    ended: bool
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None: ...
    def fetch(self, possibly_batched_index): ...

class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index): ...
