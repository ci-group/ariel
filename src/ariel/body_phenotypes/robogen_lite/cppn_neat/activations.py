import math

def sigmoid(x: int) -> float:
    """The standard logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x: int) -> float:
    """Hyperbolic tangent, maps values to (-1, 1)."""
    return math.tanh(x)

def sin_act(x: int) -> float:
    """Sine function, useful for generating repeating patterns."""
    return math.sin(x)

def gaussian(x: int) -> float:
    """Gaussian (bell curve) function."""
    return math.exp(-(x**2))

def relu(x: int) -> float:
    """Rectified Linear Unit."""
    return max(0.0, x)

ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "sin": sin_act,
    "gaussian": gaussian,
    "relu": relu,
}

DEFAULT_ACTIVATION = "sigmoid"