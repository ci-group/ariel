"""Reversible Differential Evolution (RevDE) optimizer.

Notes
-----
    *

References
----------
    [1] https://dl.acm.org/doi/abs/10.1145/3377929.3389972

Todo
----
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.traceback import install

# Type Aliases
ArrayGenotype = npt.NDArray[np.float64]

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP ---
SEED = 42
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP ---
install(show_locals=True)
console = Console()


class RevDE:
    def __init__(self, scaling_factor: float) -> None:
        # Passed parameters
        f = scaling_factor
        f2 = f**2
        f3 = f**3

        # Prep work
        a = 1 - f2
        b = f + f2
        c = -f + f2 + f3
        d = 1 - (2 * f2) - f3

        # Linear transformation matrix
        self.r_matrix = np.array([
            [1, f, -f],
            [-f, a, b],
            [b, c, d],
        ])

    def mutate(
        self,
        parent_a: ArrayGenotype,
        parent_b: ArrayGenotype,
        parent_c: ArrayGenotype,
    ) -> list[ArrayGenotype]:
        # Ensure parents have the same shape
        if not (parent_a.shape == parent_b.shape == parent_c.shape):
            msg = "Parents must have the same shape"
            raise ValueError(msg)

        # Perform mutation
        x_matrix = np.vstack((parent_a, parent_b, parent_c))
        out = self.r_matrix @ x_matrix
        y1, y2, y3 = out
        return [y1, y2, y3]


def main() -> None:
    """Entry point."""
    parent_b = RNG.random(4)
    parent_c = RNG.random(4)
    parent_a = RNG.random(4)
    revde = RevDE(scaling_factor=-0.5)
    children = revde.mutate(parent_a, parent_b, parent_c)

    # Display parents
    for i, parent in enumerate([parent_a, parent_b, parent_c]):
        console.log(f"Parent {i}: {parent}")

    # Display children
    for i, child in enumerate(children):
        console.log(f"Child {i}: {child}")


if __name__ == "__main__":
    main()
