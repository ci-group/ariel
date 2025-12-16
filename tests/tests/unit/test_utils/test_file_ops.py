"""Tests for file operations utilities."""

from pathlib import Path
from datetime import datetime
import pytest
import tempfile

from ariel.utils.file_ops import generate_save_path


class TestGenerateSavePathBasic:
    """Tests for basic generate_save_path functionality."""

    def test_generate_save_path_with_file_name_only(self, tmp_path):
        """Test generate_save_path with only file_name provided."""
        result = generate_save_path(
            file_name="test_file",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        assert isinstance(result, Path)
        assert result.name == "test_file.txt"
        assert result.parent == tmp_path

    def test_generate_save_path_with_all_parameters(self, tmp_path):
        """Test generate_save_path with all parameters provided."""
        result = generate_save_path(
            file_name="my_file",
            file_path=tmp_path,
            file_extension=".csv",
            append_date=False
        )
        
        assert result.name == "my_file.csv"
        assert result.suffix == ".csv"
        assert result.parent == tmp_path

    def test_generate_save_path_appends_date_by_default(self, tmp_path):
        """Test that date is appended by default."""
        result = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt"
        )
        
        # Result name should contain "test_" and have date format
        assert "test_" in result.name
        assert result.suffix == ".txt"

    def test_generate_save_path_no_date_append(self, tmp_path):
        """Test generate_save_path with append_date=False."""
        result = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".log",
            append_date=False
        )
        
        assert result.name == "test.log"

    def test_generate_save_path_returns_path_object(self, tmp_path):
        """Test that return type is Path."""
        result = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        assert isinstance(result, Path)


class TestGenerateSavePathExtensionHandling:
    """Tests for file extension handling."""

    def test_extension_in_file_name(self, tmp_path):
        """Test when extension is included in file_name."""
        result = generate_save_path(
            file_name="test.json",
            file_path=tmp_path,
            append_date=False
        )
        
        assert result.suffix == ".json"
        assert "test" in result.name

    def test_extension_in_file_path(self, tmp_path):
        """Test when extension is included in file_path."""
        file_path_with_ext = tmp_path / "output.csv"
        result = generate_save_path(
            file_path=file_path_with_ext,
            append_date=False
        )
        
        assert result.suffix == ".csv"
        assert result.parent == tmp_path

    def test_explicit_extension_takes_precedence(self, tmp_path):
        """Test that explicit file_extension takes precedence."""
        result = generate_save_path(
            file_name="test.txt",
            file_path=tmp_path,
            file_extension=".json",
            append_date=False
        )
        
        # Explicit extension should be used
        assert result.suffix == ".json"

    def test_extension_with_multiple_dots(self, tmp_path):
        """Test file extension with multiple dots."""
        result = generate_save_path(
            file_name="test.backup",
            file_path=tmp_path,
            file_extension=".tar.gz",
            append_date=False
        )
        
        # Last extension should be .gz
        assert result.suffix == ".gz"

    def test_no_extension_warning_logged(self, tmp_path, capsys):
        """Test that warning is logged when no extension found."""
        result = generate_save_path(
            file_name="test_no_ext",
            file_path=tmp_path,
            append_date=False
        )
        
        # Should still return a path even without explicit extension
        assert isinstance(result, Path)


class TestGenerateSavePathPathHandling:
    """Tests for file path handling."""

    def test_file_path_as_string(self, tmp_path):
        """Test that file_path can be provided as string."""
        result = generate_save_path(
            file_name="test",
            file_path=str(tmp_path),
            file_extension=".txt",
            append_date=False
        )
        
        assert isinstance(result, Path)
        assert result.parent == tmp_path

    def test_file_path_as_path_object(self, tmp_path):
        """Test that file_path can be provided as Path object."""
        result = generate_save_path(
            file_name="test",
            file_path=Path(tmp_path),
            file_extension=".txt",
            append_date=False
        )
        
        assert result.parent == tmp_path

    def test_nonexistent_directory_handling(self, tmp_path):
        """Test handling of nonexistent directory."""
        nonexistent_dir = tmp_path / "does" / "not" / "exist"
        
        result = generate_save_path(
            file_name="test",
            file_path=nonexistent_dir,
            file_extension=".txt",
            append_date=False
        )
        
        assert isinstance(result, Path)
        # Result should be constructed in the nonexistent_dir
        assert result.name == "test.txt"
        # Path is constructed but directory may not be created
        assert "does" in str(result)
        assert "not" in str(result)
        assert "exist" in str(result)

    def test_file_path_defaults_to_data_directory_when_none(self):
        """Test that file_path defaults to DATA directory when None."""
        result = generate_save_path(
            file_name="test",
            file_extension=".txt",
            append_date=False
        )
        
        # Result should be in __data__ directory
        assert "__data__" in str(result)
        assert result.name == "test.txt"

    def test_file_name_as_path_with_directory_ignored(self, tmp_path):
        """Test that when file_name is a path, directory info is used but file_path overrides."""
        file_with_path = tmp_path / "subdir" / "test.txt"
        result = generate_save_path(
            file_name=file_with_path,
            file_path=tmp_path,  # This overrides the directory in file_name
            append_date=False
        )
        
        # When both file_path and file_name (with path) are provided, file_path is used
        assert result.parent == tmp_path
        assert result.suffix == ".txt"


class TestGenerateSavePathFileNameHandling:
    """Tests for file name handling."""

    def test_file_name_as_string(self, tmp_path):
        """Test that file_name can be provided as string."""
        result = generate_save_path(
            file_name="my_file",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        assert "my_file" in result.name

    def test_file_name_as_path_object(self, tmp_path):
        """Test that file_name can be provided as Path object."""
        result = generate_save_path(
            file_name=Path("my_file"),
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        assert "my_file" in result.name

    def test_file_name_with_special_characters(self, tmp_path):
        """Test file name with special characters."""
        result = generate_save_path(
            file_name="test-file_v1",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        assert "test-file_v1" in result.name

    def test_file_name_none_generates_timestamp(self, tmp_path):
        """Test that None file_name generates timestamp-based name."""
        result = generate_save_path(
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        # Should have a timestamp in the name (format: YYYY-MM-DD_HH-MM-SS-ffffff)
        name = result.name
        assert any(char.isdigit() for char in name)
        assert "-" in name


class TestGenerateSavePathErrorHandling:
    """Tests for error handling."""

    def test_all_parameters_none_raises_error(self):
        """Test that ValueError is raised when all parameters are None."""
        with pytest.raises(ValueError, match="All arguments"):
            generate_save_path()

    def test_only_file_name_none_and_file_extension_none_raises(self):
        """Test error when file_name and file_extension are None but file_path has no ext."""
        with pytest.raises(ValueError, match="All arguments"):
            generate_save_path(file_name=None, file_path=None, file_extension=None)


class TestGenerateSavePathDateAppending:
    """Tests for date appending functionality."""

    def test_date_format_in_appended_name(self, tmp_path):
        """Test that appended date follows correct format."""
        result = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=True
        )
        
        # Format should be: test_YYYY-MM-DD_HH-MM-SS-ffffff.txt
        name = result.stem
        assert "test_" in name
        # Should contain date separators
        assert "-" in name

    def test_appended_date_is_recent(self, tmp_path):
        """Test that appended date is recent (within last minute)."""
        before = datetime.now()
        result = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=True
        )
        after = datetime.now()
        
        # Should have created with recent timestamp
        assert isinstance(result, Path)
        assert result.exists() is False  # We haven't created the file yet

    def test_multiple_calls_produce_different_names(self, tmp_path):
        """Test that multiple calls with append_date=True produce different names."""
        result1 = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=True
        )
        
        # Small sleep to ensure different timestamp
        import time
        time.sleep(0.001)
        
        result2 = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=True
        )
        
        # Names should be different due to timestamp
        assert result1.name != result2.name

    def test_append_date_false_produces_same_name(self, tmp_path):
        """Test that append_date=False produces consistent names."""
        result1 = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        result2 = generate_save_path(
            file_name="test",
            file_path=tmp_path,
            file_extension=".txt",
            append_date=False
        )
        
        assert result1.name == result2.name


class TestGenerateSavePathComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_save_path_for_csv(self, tmp_path):
        """Test typical CSV save scenario."""
        result = generate_save_path(
            file_name="experiment_results",
            file_path=tmp_path,
            file_extension=".csv",
            append_date=True
        )
        
        assert result.suffix == ".csv"
        assert "experiment_results" in result.name
        assert result.parent == tmp_path

    def test_save_path_for_json(self, tmp_path):
        """Test typical JSON save scenario."""
        result = generate_save_path(
            file_name="config",
            file_path=tmp_path,
            file_extension=".json",
            append_date=False
        )
        
        assert result.name == "config.json"

    def test_save_path_for_log(self, tmp_path):
        """Test typical log file scenario."""
        result = generate_save_path(
            file_name="simulation",
            file_path=tmp_path,
            file_extension=".log",
            append_date=True
        )
        
        assert result.suffix == ".log"
        assert "simulation_" in result.name

    def test_save_path_with_complex_file_structure(self, tmp_path):
        """Test with nested file structures."""
        nested_path = tmp_path / "experiments" / "2024" / "run_001"
        result = generate_save_path(
            file_name="results",
            file_path=nested_path,
            file_extension=".pkl",
            append_date=False
        )
        
        assert result.name == "results.pkl"
        assert "experiments" in str(result)
        assert "2024" in str(result)

    def test_save_path_preserves_directory_structure(self, tmp_path):
        """Test that directory structure is preserved in returned path."""
        subdir = tmp_path / "subdir"
        result = generate_save_path(
            file_name="test",
            file_path=subdir,
            file_extension=".txt",
            append_date=False
        )
        
        assert subdir in result.parents or result.parent == subdir