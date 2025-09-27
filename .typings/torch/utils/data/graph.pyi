from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe

__all__ = ['traverse', 'traverse_dps']

DataPipe = IterDataPipe | MapDataPipe

def traverse_dps(datapipe: DataPipe) -> DataPipeGraph:
    """
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    This only looks into the attribute from each DataPipe that is either a
    DataPipe and a Python collection object such as ``list``, ``tuple``,
    ``set`` and ``dict``.

    Args:
        datapipe: the end DataPipe of the graph
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
def traverse(datapipe: DataPipe, only_datapipe: bool | None = None) -> DataPipeGraph:
    """
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    [Deprecated]
    When ``only_dataPipe`` is specified as ``True``, it would only look into the
    attribute from each DataPipe that is either a DataPipe and a Python collection object
    such as ``list``, ``tuple``, ``set`` and ``dict``.

    Note:
        This function is deprecated. Please use `traverse_dps` instead.

    Args:
        datapipe: the end DataPipe of the graph
        only_datapipe: If ``False`` (default), all attributes of each DataPipe are traversed.
          This argument is deprecating and will be removed after the next release.
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
