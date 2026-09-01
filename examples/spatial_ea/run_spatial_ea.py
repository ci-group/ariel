r"""Run one spatial evolutionary algorithm from the command line.

A thin driver over :mod:`ariel.spatial_ea.cli`, kept here so the example is
discoverable next to the other ARIEL examples. Equivalent to running
``python -m ariel.spatial_ea``.

Examples
--------
Short smoke run with everything on the defaults::

    python examples/spatial_ea/run_spatial_ea.py \\
        --population-size 6 --num-generations 3 --simulation-time 1.0

Reproduce a research-prototype configuration::

    python examples/spatial_ea/run_spatial_ea.py \\
        --config examples/spatial_ea/ea_config.yaml

"""

# Standard library
import sys

# Local libraries
from ariel.spatial_ea.cli import main

if __name__ == "__main__":
    sys.exit(main())
