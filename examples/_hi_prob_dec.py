"""Example of high-probability-decoding for graphs.

Author:     jmdm
Date:       2025-06-25
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     Completed ✅
"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import numpy as np
from rich.console import Console

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    draw_graph,
    save_graph_as_json,
)

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def main() -> None:
    """Entry point."""
    # System parameters
    num_modules = 30
    console.log(f"Number of modules: {num_modules}")

    # "Type" probability space
    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = RNG.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    rotation_probability_space = RNG.random(
        size=(num_modules, NUM_OF_ROTATIONS),
        dtype=np.float32,
    )

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )

    # Visualize the graph
    draw_graph(graph)

    # Save the graph to a file
    save_graph_as_json(
        graph,
        DATA / "graph.json",
    )


if __name__ == "__main__":
    main()
