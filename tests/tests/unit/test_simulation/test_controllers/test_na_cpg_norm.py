"""Not ready yet."""

import importlib
import importlib.util
import sys
from pathlib import Path
import pytest

MODULE_PATH = Path("src/ariel/simulation/controllers/na_cpg_norm.py")


def test_importing_module_raises_not_implemented():
    """The NaCPG_Norm class is a work-in-progress and should raise NotImplementedError on import."""
    # Importing the package module should raise the deliberate NotImplementedError
    with pytest.raises(NotImplementedError):
        importlib.import_module("ariel.simulation.controllers.na_cpg_norm")


def test_source_file_contains_expected_symbols():
    """Sanity-check the source file contains the adjacency helper and the class declaration."""
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "def create_fully_connected_adjacency" in src
    assert "class NaCPG_Norm" in src
    assert "NotImplementedError" in src