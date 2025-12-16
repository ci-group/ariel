import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def ensure_local_package(monkeypatch):
    """
    Ensure tests import the local `src` copy of the package when running in the repo.
    If you installed the package editable (`pip install -e .`) this is a no-op.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    if src.exists():
        # Prepend src to sys.path so imports resolve to local code
        import sys
        sys.path.insert(0, str(src))
    yield