import torch
from torch.utils.data.graph import DataPipe, DataPipeGraph
from typing import Any

__all__ = ['apply_random_seed', 'apply_sharding', 'apply_shuffle_seed', 'apply_shuffle_settings', 'get_all_graph_pipes']

def get_all_graph_pipes(graph: DataPipeGraph) -> list[DataPipe]: ...
def apply_sharding(datapipe: DataPipe, num_of_instances: int, instance_id: int, sharding_group=...) -> DataPipe:
    """
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.

    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
def apply_shuffle_settings(datapipe: DataPipe, shuffle: bool | None = None) -> DataPipe:
    """
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute.

    Apply the method to each `DataPipe` that has APIs of ``set_shuffle``
    and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe: ...
def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe:
    """
    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of ``set_seed``.

    Then set the random seed based on the provided RNG to those ``DataPipe``.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    """
