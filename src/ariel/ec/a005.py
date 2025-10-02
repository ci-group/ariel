"""TODO(jmdm): description of script."""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.ec.a000 import IntegersGenerator
from ariel.ec.a001 import JSONIterable

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)
DB_NAME = "database.db"
DB_PATH: Path = DATA / DB_NAME
SEED = 42

# Global functions
install(width=180)
console = Console()
RNG = np.random.default_rng(SEED)


class Crossover:
    @staticmethod
    def one_point(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
    ) -> tuple[JSONIterable, JSONIterable]:
        # Prep
        parent_i_arr_shape = np.array(parent_i).shape
        parent_j_arr_shape = np.array(parent_j).shape
        parent_i_arr = np.array(parent_i).flatten().copy()
        parent_j_arr = np.array(parent_j).flatten().copy()

        # Ensure parents have the same length
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)

        # Select crossover point
        crossover_point = RNG.integers(0, len(parent_i_arr))

        # Copy over parents
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()

        # Perform crossover
        child1[crossover_point:] = parent_j_arr[crossover_point:]
        child2[crossover_point:] = parent_i_arr[crossover_point:]

        # Correct final shape
        child1 = child1.reshape(parent_i_arr_shape).astype(int).tolist()
        child2 = child2.reshape(parent_j_arr_shape).astype(int).tolist()
        return child1, child2


def main() -> None:
    """Entry point."""
    p1 = IntegersGenerator.integers(-5, 5, (2, 5))
    p2 = IntegersGenerator.choice([1, 3, 4], (2, 5))
    console.log(p1, p2)

    c1, c2 = Crossover.one_point(p1, p2)
    console.log(c1, c2)


if __name__ == "__main__":
    main()
