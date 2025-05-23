"""Tests for the Pattern and PatternLibrary classes."""

import tempfile
import json
from pathlib import Path

from cellular.core.grid import Grid
from cellular.core.patterns import Pattern, PatternLibrary


class TestPattern:
    """Test cases for the Pattern class."""

    def test_initialization(self):
        """Test pattern initialization."""
        cells = [(0, 0), (1, 0), (2, 0)]
        pattern = Pattern("Blinker", cells, "Period-2 oscillator")

        assert pattern.name == "Blinker"
        assert pattern.cells == cells
        assert pattern.description == "Period-2 oscillator"
        assert pattern.metadata == {}

    def test_initialization_with_metadata(self):
        """Test pattern initialization with metadata."""
        cells = [(0, 0), (1, 1)]
        metadata = {"period": 2, "type": "oscillator"}
        pattern = Pattern("Test", cells, metadata=metadata)

        assert pattern.metadata == metadata

    def test_apply_to_grid(self):
        """Test applying pattern to grid."""
        grid = Grid(10, 10)
        cells = [(0, 0), (1, 0), (2, 0)]
        pattern = Pattern("Blinker", cells)

        pattern.apply_to_grid(grid)

        assert grid.get_cell(0, 0)
        assert grid.get_cell(1, 0)
        assert grid.get_cell(2, 0)
        assert not grid.get_cell(0, 1)
        assert grid.population == 3

    def test_apply_to_grid_with_offset(self):
        """Test applying pattern with offset."""
        grid = Grid(10, 10)
        cells = [(0, 0), (1, 0), (2, 0)]
        pattern = Pattern("Blinker", cells)

        pattern.apply_to_grid(grid, offset_x=5, offset_y=3)

        assert grid.get_cell(5, 3)
        assert grid.get_cell(6, 3)
        assert grid.get_cell(7, 3)
        assert not grid.get_cell(0, 0)
        assert grid.population == 3

    def test_apply_to_grid_out_of_bounds(self):
        """Test applying pattern that goes out of bounds."""
        grid = Grid(3, 3, wrap_edges=False)
        cells = [(0, 0), (1, 0), (2, 0), (3, 0)]  # Last cell out of bounds
        pattern = Pattern("Test", cells)

        # Should not raise error, just skip out-of-bounds cells
        pattern.apply_to_grid(grid)

        assert grid.get_cell(0, 0)
        assert grid.get_cell(1, 0)
        assert grid.get_cell(2, 0)
        assert grid.population == 3  # Fourth cell skipped

    def test_get_bounding_box(self):
        """Test bounding box calculation."""
        # Empty pattern
        pattern = Pattern("Empty", [])
        assert pattern.get_bounding_box() == (0, 0, 0, 0)

        # Single cell
        pattern = Pattern("Single", [(5, 3)])
        assert pattern.get_bounding_box() == (5, 3, 5, 3)

        # Multiple cells
        cells = [(1, 2), (3, 1), (0, 4), (2, 0)]
        pattern = Pattern("Multi", cells)
        assert pattern.get_bounding_box() == (0, 0, 3, 4)

    def test_get_size(self):
        """Test pattern size calculation."""
        # Empty pattern
        pattern = Pattern("Empty", [])
        assert pattern.get_size() == (1, 1)

        # Single cell
        pattern = Pattern("Single", [(5, 3)])
        assert pattern.get_size() == (1, 1)

        # 3x2 pattern
        cells = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)]
        pattern = Pattern("Rectangle", cells)
        assert pattern.get_size() == (3, 2)

    def test_normalize(self):
        """Test pattern normalization."""
        # Pattern starting at origin
        cells = [(0, 0), (1, 0), (2, 0)]
        pattern = Pattern("Test", cells)
        normalized = pattern.normalize()
        assert normalized.cells == cells
        assert normalized.name == pattern.name

        # Pattern with offset
        cells = [(5, 3), (6, 3), (7, 3)]
        pattern = Pattern("Offset", cells)
        normalized = pattern.normalize()
        assert normalized.cells == [(0, 0), (1, 0), (2, 0)]

        # Empty pattern
        pattern = Pattern("Empty", [])
        normalized = pattern.normalize()
        assert normalized.cells == []

    def test_to_dict(self):
        """Test dictionary serialization."""
        cells = [(0, 0), (1, 0)]
        metadata = {"type": "test"}
        pattern = Pattern("Test", cells, "Description", metadata)

        data = pattern.to_dict()

        assert data["name"] == "Test"
        assert data["cells"] == cells
        assert data["description"] == "Description"
        assert data["metadata"] == metadata

    def test_from_dict(self):
        """Test pattern creation from dictionary."""
        data = {
            "name": "Test",
            "cells": [(0, 0), (1, 0)],
            "description": "Description",
            "metadata": {"type": "test"},
        }

        pattern = Pattern.from_dict(data)

        assert pattern.name == "Test"
        assert pattern.cells == [(0, 0), (1, 0)]
        assert pattern.description == "Description"
        assert pattern.metadata == {"type": "test"}

    def test_from_dict_minimal(self):
        """Test pattern creation from minimal dictionary."""
        data = {"name": "Minimal", "cells": [(1, 1)]}

        pattern = Pattern.from_dict(data)

        assert pattern.name == "Minimal"
        assert pattern.cells == [(1, 1)]
        assert pattern.description == ""
        assert pattern.metadata == {}

    def test_from_grid(self):
        """Test pattern creation from grid."""
        grid = Grid(5, 5)
        grid.set_cell(1, 1, True)
        grid.set_cell(2, 1, True)
        grid.set_cell(3, 1, True)

        pattern = Pattern.from_grid(grid, "From Grid", "Test pattern")

        assert pattern.name == "From Grid"
        assert pattern.description == "Test pattern"
        assert len(pattern.cells) == 3
        assert (1, 1) in pattern.cells
        assert (2, 1) in pattern.cells
        assert (3, 1) in pattern.cells
        assert pattern.metadata["source_grid_size"] == (5, 5)
        assert pattern.metadata["population"] == 3


class TestPatternLibrary:
    """Test cases for the PatternLibrary class."""

    def test_initialization(self):
        """Test library initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(tmpdir)
            assert library.storage_dir == Path(tmpdir)

            # Should have built-in patterns
            patterns = library.list_patterns()
            assert "Block" in patterns
            assert "Blinker" in patterns
            assert "Glider" in patterns

    def test_builtin_patterns(self):
        """Test that built-in patterns are loaded correctly."""
        library = PatternLibrary()

        # Test a few key patterns
        block = library.get_pattern("Block")
        assert block is not None
        assert len(block.cells) == 4

        blinker = library.get_pattern("Blinker")
        assert blinker is not None
        assert len(blinker.cells) == 3

        glider = library.get_pattern("Glider")
        assert glider is not None
        assert len(glider.cells) == 5

    def test_add_and_get_pattern(self):
        """Test adding and retrieving patterns."""
        library = PatternLibrary()

        # Add custom pattern
        cells = [(0, 0), (1, 1), (2, 2)]
        pattern = Pattern("Diagonal", cells, "Test diagonal")
        library.add_pattern(pattern)

        # Retrieve pattern
        retrieved = library.get_pattern("Diagonal")
        assert retrieved is not None
        assert retrieved.name == "Diagonal"
        assert retrieved.cells == cells
        assert retrieved.description == "Test diagonal"

        # Non-existent pattern
        assert library.get_pattern("NonExistent") is None

    def test_list_patterns(self):
        """Test listing all patterns."""
        library = PatternLibrary()

        initial_count = len(library.list_patterns())

        # Add custom pattern
        pattern = Pattern("Custom", [(0, 0)])
        library.add_pattern(pattern)

        patterns = library.list_patterns()
        assert len(patterns) == initial_count + 1
        assert "Custom" in patterns

    def test_get_patterns_by_category(self):
        """Test pattern categorization."""
        library = PatternLibrary()

        categories = library.get_patterns_by_category()

        # Check expected categories exist
        assert "Still Life" in categories
        assert "Oscillators" in categories
        assert "Spaceships" in categories
        assert "Methuselahs" in categories

        # Check some patterns are in correct categories
        assert "Block" in categories["Still Life"]
        assert "Blinker" in categories["Oscillators"]
        assert "Glider" in categories["Spaceships"]
        assert "R-pentomino" in categories["Methuselahs"]

        # Add custom pattern
        pattern = Pattern("Custom", [(0, 0)])
        library.add_pattern(pattern)

        categories = library.get_patterns_by_category()
        assert "Custom" in categories
        assert "Custom" in categories["Custom"]

    def test_save_and_load_pattern(self):
        """Test saving and loading patterns to/from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(tmpdir)

            # Create and save pattern
            cells = [(0, 0), (1, 0), (0, 1)]
            pattern = Pattern("TestSave", cells, "Test save/load")
            library.save_pattern(pattern, "test_pattern.json")

            # Check file exists
            file_path = Path(tmpdir) / "test_pattern.json"
            assert file_path.exists()

            # Load pattern
            library2 = PatternLibrary(tmpdir)
            loaded_pattern = library2.load_pattern("test_pattern.json")

            assert loaded_pattern.name == "TestSave"
            assert loaded_pattern.cells == cells
            assert loaded_pattern.description == "Test save/load"

            # Check it's in the library
            assert library2.get_pattern("TestSave") is not None

    def test_save_pattern_default_filename(self):
        """Test saving pattern with default filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(tmpdir)

            pattern = Pattern("Test Pattern", [(0, 0)])
            library.save_pattern(pattern)

            # Should create file with sanitized name
            expected_file = Path(tmpdir) / "test_pattern.json"
            assert expected_file.exists()

    def test_load_all_patterns(self):
        """Test loading all patterns from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(tmpdir)

            # Create some pattern files
            pattern1_data = {
                "name": "Pattern1",
                "cells": [(0, 0)],
                "description": "First pattern",
                "metadata": {},
            }
            pattern2_data = {
                "name": "Pattern2",
                "cells": [(1, 1)],
                "description": "Second pattern",
                "metadata": {},
            }

            with open(Path(tmpdir) / "pattern1.json", "w") as f:
                json.dump(pattern1_data, f)
            with open(Path(tmpdir) / "pattern2.json", "w") as f:
                json.dump(pattern2_data, f)

            # Load all patterns
            library.load_all_patterns()

            # Check they're loaded
            assert library.get_pattern("Pattern1") is not None
            assert library.get_pattern("Pattern2") is not None

    def test_load_invalid_pattern_file(self):
        """Test loading invalid pattern file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(tmpdir)

            # Create invalid file
            with open(Path(tmpdir) / "invalid.json", "w") as f:
                f.write("invalid json")

            # Should not raise exception, just print warning
            library.load_all_patterns()

    def test_save_grid_as_pattern(self):
        """Test saving grid state as pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(tmpdir)

            # Create grid with pattern
            grid = Grid(5, 5)
            grid.set_cell(1, 1, True)
            grid.set_cell(2, 1, True)
            grid.set_cell(3, 1, True)

            # Save as pattern
            pattern = library.save_grid_as_pattern(grid, "Grid Pattern", "From grid", "grid_pattern.json")

            # Check pattern was created
            assert pattern.name == "Grid Pattern"
            assert pattern.description == "From grid"
            assert len(pattern.cells) == 3
            assert pattern.metadata["source_grid_size"] == (5, 5)
            assert pattern.metadata["population"] == 3

            # Check it's in library
            assert library.get_pattern("Grid Pattern") is not None

            # Check file was saved
            assert (Path(tmpdir) / "grid_pattern.json").exists()

    def test_save_grid_as_pattern_no_file(self):
        """Test saving grid as pattern without file save."""
        library = PatternLibrary()

        grid = Grid(3, 3)
        grid.set_cell(1, 1, True)

        pattern = library.save_grid_as_pattern(grid, "Test", "No file")

        # Should be in library but not saved to disk
        assert library.get_pattern("Test") is not None
        assert pattern.name == "Test"
