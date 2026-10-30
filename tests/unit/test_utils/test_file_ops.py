"""Test: generate_save_path file-path generation utility."""

import re
from pathlib import Path

import pytest

from ariel.utils.file_ops import generate_save_path


def test_generate_save_path_with_name_and_extension(tmp_path) -> None:
    """file_name + file_extension produces a path in the default DATA dir."""
    result = generate_save_path(
        file_name="my_file",
        file_extension=".csv",
        file_path=tmp_path,
        append_date=False,
    )
    assert isinstance(result, Path)
    assert result.suffix == ".csv"
    assert result.stem == "my_file"


def test_generate_save_path_appends_date(tmp_path) -> None:
    """append_date=True inserts a timestamp into the file name."""
    result = generate_save_path(
        file_name="run",
        file_extension=".log",
        file_path=tmp_path,
        append_date=True,
    )
    # timestamp pattern: YYYY-MM-DD_HH-MM-SS-microseconds
    pattern = r"run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+\.log"
    assert re.search(pattern, result.name), f"Unexpected name: {result.name}"


def test_generate_save_path_no_date(tmp_path) -> None:
    """append_date=False keeps the stem clean."""
    result = generate_save_path(
        file_name="output",
        file_extension=".txt",
        file_path=tmp_path,
        append_date=False,
    )
    assert result.name == "output.txt"


def test_generate_save_path_extension_from_file_path(tmp_path) -> None:
    """Extension can be inferred from a file_path with suffix."""
    result = generate_save_path(
        file_path=tmp_path / "data.json",
        append_date=False,
    )
    assert result.suffix == ".json"


def test_generate_save_path_extension_from_file_name(tmp_path) -> None:
    """Extension can be inferred from a file_name with suffix."""
    result = generate_save_path(
        file_name="results.csv",
        file_path=tmp_path,
        append_date=False,
    )
    assert result.suffix == ".csv"


def test_generate_save_path_all_none_raises() -> None:
    """Calling with all None arguments raises ValueError."""
    with pytest.raises(ValueError):
        generate_save_path()


def test_generate_save_path_returns_path_object(tmp_path) -> None:
    """Return type is always a pathlib.Path."""
    result = generate_save_path(
        file_name="test",
        file_extension=".bin",
        file_path=tmp_path,
    )
    assert isinstance(result, Path)
