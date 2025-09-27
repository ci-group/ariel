import numpy as np
import typing as tp
from _typeshed import Incomplete

class VectorNode:
    """A node object of the VectorLinkedList.
    A VectorNode is a point in a space with dim = `dimension`, and an optional
    `coordinate` (which can be assigned after the VectorNode initialization).
    The VectorNode object points to two arrays with self.next and self.prev attributes.
    The `self.next` contains a list of VectorNode (aka geometric points), such that
    `self.next[i]` is a VectorNode immediately after the `self` on the i-th coordinate,
    `self.prev[j]` is a VectorNode immediately before the `self` on the j-th coordinate.
    The `area` is a vector, with its i-th element equal the area of the projection
    of the `coordinate` on the (i-1) subspace.
    The `volume` is the product of the `area` by the difference between the i-th
    coordinate of the self and self.prev[i].
    The `dominated_flag` is used to skip dominated points (see section III.C).
    The VectorNode data structure is introduced in section III.A of the original paper..
    """
    dimension: Incomplete
    coordinates: Incomplete
    _next: list['VectorNode']
    _prev: list['VectorNode']
    dominated_flag: int
    area: Incomplete
    volume: Incomplete
    def __init__(self, dimension: int, coordinates: np.ndarray | list[float] | None = None) -> None: ...
    def __str__(self) -> str: ...
    def __lt__(self, other: tp.Any) -> bool: ...
    def configure_area(self, dimension: int) -> None: ...
    @property
    def next(self) -> list['VectorNode']: ...
    @property
    def prev(self) -> list['VectorNode']: ...
    def pop(self, index: int) -> None:
        """Assigns the references of the self predecessor and successor at
        `index` index to each other, removes the links to the `self` node.
        """

class VectorLinkedList:
    """Linked list structure with list of VectorNodes as elements."""
    dimension: Incomplete
    sentinel: Incomplete
    def __init__(self, dimension: int) -> None: ...
    @classmethod
    def create_sorted(cls, dimension: int, points: tp.Any) -> VectorLinkedList:
        """Instantiate a VectorLinkedList of dimension `dimension`. The list is
        populated by nodes::VectorNode created from `points`. The nodes are sorted
        by i-th coordinate attribute in i-th row."""
    @staticmethod
    def sort_by_index(node_list: list[VectorNode], dimension_index: int) -> list[VectorNode]:
        """Returns a sorted list of `VectorNode`, with the sorting key defined by the
        `dimension_index`-th coordinates of the nodes in the `node_list`."""
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def chain_length(self, index: int) -> int: ...
    def append(self, node: VectorNode, index: int) -> None:
        """Append a node to the `index`-th position."""
    def extend(self, nodes: list[VectorNode], index: int) -> None:
        """Extends the VectorLinkedList with a list of nodes
        at `index` position"""
    @staticmethod
    def update_coordinate_bounds(bounds: np.ndarray, node: VectorNode, index: int) -> np.ndarray: ...
    def pop(self, node: VectorNode, index: int) -> VectorNode:
        """Removes and returns 'node' from all lists at the
        positions from 0 in index (exclusively)."""
    def reinsert(self, node: VectorNode, index: int) -> None:
        """
        Inserts 'node' at the position it had before it was removed
        in all lists at the positions from 0 in index (exclusively).
        This method assumes that the next and previous nodes of the
        node that is reinserted are in the list.
        """
    def iterate(self, index: int, start: VectorNode | None = None) -> tp.Iterator[VectorNode]: ...
    def reverse_iterate(self, index: int, start: VectorNode | None = None) -> tp.Iterator[VectorNode]: ...

class HypervolumeIndicator:
    '''Core class to calculate the hypervolme value of a set of points.
    As introduced in the original paper, "the indicator is a measure of
    the region which is simultaneously dominated by a set of points P,
    and bounded by a reference point r = `self.reference_bounds`. It is
    a union of axis-aligned hyper-rectangles with one common vertex, r."

    To calculate the hypervolume indicator, initialize an instance of the
    HypervolumeIndicator; the hypervolume of a set of points P is calculated
    by HypervolumeIndicator.compute(points = P) method.

    For the algorithm, refer to the section III and Algorithms 2, 3 of the
    paper `An Improved Dimension-Sweep Algorithm for the Hypervolume Indicator`
    by C.M. Fonseca et all, IEEE Congress on Evolutionary Computation, 2006.
    '''
    reference_point: Incomplete
    dimension: Incomplete
    reference_bounds: Incomplete
    _multilist: VectorLinkedList | None
    def __init__(self, reference_point: np.ndarray) -> None: ...
    @property
    def multilist(self) -> VectorLinkedList: ...
    def compute(self, points: list[np.ndarray] | np.ndarray) -> float: ...
    def plane_hypervolume(self) -> float:
        """Calculates the hypervolume on a two dimensional plane. The algorithm
        is described in Section III-A of the original paper."""
    def recursive_hypervolume(self, dimension: int) -> float:
        """Recursive hypervolume computation. The algorithm is provided by Algorithm 3.
        of the original paper."""
    def skip_dominated_points(self, node: VectorNode, dimension: int) -> None:
        """Implements Algorithm 2, _skipdom_, for skipping dominated points."""
