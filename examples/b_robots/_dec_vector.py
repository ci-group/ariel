"""Example of vector-decoding for graphs."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import numpy as np
from rich.console import Console

# Local libraries
from ariel.body_phenotypes.robogen_lite.decoders import (
    VectorDecoder,
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
DATA.mkdir(exist_ok=True, parents=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def main() -> None:
    """Entry point."""
    # System parameters
    num_modules = 30
    console.log(f"Number of modules: {num_modules}")

    # "Type" vector
    type_vector = RNG.random(
        size=(num_modules,),
        dtype=np.float32,
    )

    # "Connection" vector
    conn_vector = RNG.random(
        size=(num_modules, num_modules),
        dtype=np.float32,
    )

    # "Rotation" vector
    rotation_vector = RNG.random(
        size=(num_modules,),
        dtype=np.float32,
    )

    # Decode the high-probability graph
    hpd = VectorDecoder(num_modules)
    graph: DiGraph[Any] = hpd.vectors_to_graph(
        type_vector=type_vector,
        connection_vector=conn_vector,
        rotation_vector=rotation_vector,
    )

    # Visualize the graph
    draw_graph(graph)

    # Save the graph to a file
    # save_graph_as_json(
    #     graph,
    #     DATA / "graph.json",
    # )


if __name__ == "__main__":
    main()
