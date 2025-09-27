import torch

def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (Graph): The graph to be annotated
    """
