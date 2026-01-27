from typing import Callable
ActivationFunction = Callable[[float], float]

class Node:
    """A node gene in a NEAT genome."""
    def __init__(self,
                  _id: int,
                  typ: str, 
                  activation: ActivationFunction,
                  bias: float
                ):
        self._id = _id
        self.typ = typ
        self.activation = activation
        self.bias = bias

    def copy(self):
        """Returns a new Node object with identical values."""
        return Node(
            self._id, 
            self.typ, 
            self.activation, 
            self.bias
        )

    