import abc
from _typeshed import Incomplete
from torch.fx.graph_module import GraphModule
from typing import NamedTuple

__all__ = ['PassResult', 'PassBase']

class PassResult(NamedTuple('PassResult', [('graph_module', Incomplete), ('modified', Incomplete)])):
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """
    __slots__: Incomplete
    def __new__(cls, graph_module, modified): ...

class PassBase(abc.ABC, metaclass=abc.ABCMeta):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """
    def __call__(self, graph_module: GraphModule) -> PassResult | None:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """
    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> PassResult | None:
        """
        The pass that is run through the given graph module. To implement a
        pass, it is required to implement this function.

        Args:
            graph_module: The graph module we will run a pass on
        """
    def requires(self, graph_module: GraphModule) -> None:
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
    def ensures(self, graph_module: GraphModule) -> None:
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
