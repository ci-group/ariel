"""
ARIEL: Autonomous Robots through Integrated Evolution and Learning.

Todo
----
    [ ] Make rich logger take command-line argument for verbosity  (click?)
"""

# Standard library
import logging
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# --- DATA SETUP --- #
CWD = Path.cwd()
DATA = Path(CWD / "__data__")

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP --- #
FORMAT = "%(message)s"
install()
console = Console()
logging.basicConfig(
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging_level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
log = logging.getLogger("rich")
log.setLevel(logging_level)
