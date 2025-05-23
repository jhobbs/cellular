"""Tests for the Grid class."""

import pytest
from cellular.core.grid import Grid


class TestGrid:
    """Test cases for the Grid class."""

    def test_initialization(self):
        """Test grid initialization."""
        grid = Grid(10, 20)
        assert grid.width == 10
        assert grid.height == 20
        assert grid.shape == (10, 20)
        assert grid.wrap_edges is True
        assert grid.population == 0

    def test_initialization_no_wrap(self):
        """Test grid initialization without edge wrapping."""
        grid = Grid(5, 5, wrap_edges=False)
        assert grid.wrap_edges is False

    def test_cell_operations(self):
        """Test basic cell get/set operations."""
        grid = Grid(5, 5)

        # Initially all cells should be dead
        assert not grid.get_cell(0, 0)
        assert not grid.get_cell(2, 3)

        # Set some cells alive
        grid.set_cell(1, 1, True)
        grid.set_cell(2, 3, True)

        assert grid.get_cell(1, 1)
        assert grid.get_cell(2, 3)
        assert not grid.get_cell(0, 0)

        # Set cell dead
        grid.set_cell(1, 1, False)
        assert not grid.get_cell(1, 1)

    def test_toggle_cell(self):
        """Test cell toggling."""
        grid = Grid(5, 5)

        # Toggle dead cell to alive
        result = grid.toggle_cell(2, 2)
        assert result is True
        assert grid.get_cell(2, 2) is True

        # Toggle alive cell to dead
        result = grid.toggle_cell(2, 2)
        assert result is False
        assert grid.get_cell(2, 2) is False

    def test_wrap_edges(self):
        """Test edge wrapping behavior."""
        grid = Grid(3, 3, wrap_edges=True)

        # Test wrapping
        grid.set_cell(-1, -1, True)  # Should wrap to (2, 2)
        assert grid.get_cell(2, 2)

        grid.set_cell(3, 4, True)  # Should wrap to (0, 1)
        assert grid.get_cell(0, 1)

        # Test getting with wrapping
        assert grid.get_cell(-1, -1)  # Should return (2, 2)
        assert grid.get_cell(3, 4)  # Should return (0, 1)

    def test_no_wrap_edges(self):
        """Test behavior without edge wrapping."""
        grid = Grid(3, 3, wrap_edges=False)

        # These should raise IndexError
        with pytest.raises(IndexError):
            grid.set_cell(-1, 0, True)

        with pytest.raises(IndexError):
            grid.set_cell(0, -1, True)

        with pytest.raises(IndexError):
            grid.set_cell(3, 0, True)

        with pytest.raises(IndexError):
            grid.set_cell(0, 3, True)

        with pytest.raises(IndexError):
            grid.get_cell(-1, 0)

        with pytest.raises(IndexError):
            grid.get_cell(3, 0)

    def test_clear(self):
        """Test grid clearing."""
        grid = Grid(5, 5)

        # Set some cells
        grid.set_cell(1, 1, True)
        grid.set_cell(2, 2, True)
        grid.set_cell(3, 3, True)
        assert grid.population == 3

        # Clear grid
        grid.clear()
        assert grid.population == 0
        assert not grid.get_cell(1, 1)
        assert not grid.get_cell(2, 2)
        assert not grid.get_cell(3, 3)

    def test_randomize(self):
        """Test random population."""
        grid = Grid(10, 10)

        # Test with probability 0 (should be empty)
        grid.randomize(0.0)
        assert grid.population == 0

        # Test with probability 1 (should be full)
        grid.randomize(1.0)
        assert grid.population == 100

        # Test with intermediate probability
        grid.randomize(0.5)
        # Should have roughly half the cells (allow some variance)
        assert 30 <= grid.population <= 70

    def test_population(self):
        """Test population counting."""
        grid = Grid(5, 5)
        assert grid.population == 0

        grid.set_cell(0, 0, True)
        assert grid.population == 1

        grid.set_cell(1, 1, True)
        grid.set_cell(2, 2, True)
        assert grid.population == 3

        grid.set_cell(0, 0, False)
        assert grid.population == 2

    def test_get_neighbors(self):
        """Test neighbor counting for individual cells."""
        grid = Grid(5, 5, wrap_edges=False)

        # Empty grid
        assert grid.get_neighbors(2, 2) == 0

        # Set up a pattern
        grid.set_cell(1, 1, True)
        grid.set_cell(1, 2, True)
        grid.set_cell(2, 1, True)

        # Test neighbor counts
        assert grid.get_neighbors(0, 0) == 1  # Only (1,1) neighbor
        assert grid.get_neighbors(2, 2) == 3  # All three cells are neighbors
        assert grid.get_neighbors(1, 1) == 2  # (1,2) and (2,1) neighbors, cell itself doesn't count
        assert grid.get_neighbors(3, 3) == 0  # No neighbors

    def test_get_neighbors_with_wrap(self):
        """Test neighbor counting with edge wrapping."""
        grid = Grid(3, 3, wrap_edges=True)

        # Set corners
        grid.set_cell(0, 0, True)
        grid.set_cell(2, 2, True)

        # Corner (0,0) should see (2,2) as neighbor due to wrapping
        neighbors = grid.get_neighbors(0, 0)
        assert neighbors == 1

        # Corner (2,2) should see (0,0) as neighbor due to wrapping
        neighbors = grid.get_neighbors(2, 2)
        assert neighbors == 1

    def test_count_all_neighbors(self):
        """Test vectorized neighbor counting."""
        grid = Grid(5, 5)

        # Set up a simple pattern
        grid.set_cell(2, 1, True)
        grid.set_cell(2, 2, True)
        grid.set_cell(2, 3, True)  # Vertical line

        neighbor_counts = grid.count_all_neighbors()

        # Check some specific positions
        assert neighbor_counts[2, 2] == 2  # Middle of line has 2 neighbors
        assert neighbor_counts[1, 2] == 3  # Next to middle has 3 neighbors
        assert neighbor_counts[3, 2] == 3  # Other side has 3 neighbors
        assert neighbor_counts[0, 0] == 0  # Far corner has no neighbors

    def test_save_and_get_changed_cells(self):
        """Test state saving and change detection."""
        grid = Grid(5, 5)

        # Set initial state
        grid.set_cell(1, 1, True)
        grid.set_cell(2, 2, True)
        grid.save_state()

        # Make changes
        grid.set_cell(1, 1, False)  # Turn off
        grid.set_cell(3, 3, True)  # Turn on
        # (2, 2) unchanged

        # Get changed cells
        changed = list(grid.get_changed_cells())
        assert len(changed) == 2
        assert (1, 1) in changed
        assert (3, 3) in changed
        assert (2, 2) not in changed

    def test_copy_from(self):
        """Test copying from another grid."""
        grid1 = Grid(3, 3)
        grid2 = Grid(3, 3)

        # Set up grid1
        grid1.set_cell(0, 0, True)
        grid1.set_cell(1, 1, True)
        grid1.set_cell(2, 2, True)

        # Copy to grid2
        grid2.copy_from(grid1)

        # Check that grid2 matches grid1
        assert grid2.get_cell(0, 0)
        assert grid2.get_cell(1, 1)
        assert grid2.get_cell(2, 2)
        assert grid2.population == 3

        # Test size mismatch
        grid3 = Grid(4, 4)
        with pytest.raises(ValueError):
            grid3.copy_from(grid1)

    def test_to_list_and_from_list(self):
        """Test serialization to/from lists."""
        grid = Grid(3, 3)

        # Set up a pattern
        grid.set_cell(0, 0, True)
        grid.set_cell(1, 1, True)
        grid.set_cell(2, 2, True)

        # Convert to list
        data = grid.to_list()
        assert len(data) == 3
        assert len(data[0]) == 3
        assert data[0][0] == 1
        assert data[1][1] == 1
        assert data[2][2] == 1
        assert data[0][1] == 0

        # Create new grid and load from list
        grid2 = Grid(3, 3)
        grid2.from_list(data)

        # Check that grids match
        assert grid2.get_cell(0, 0)
        assert grid2.get_cell(1, 1)
        assert grid2.get_cell(2, 2)
        assert not grid2.get_cell(0, 1)
        assert grid2.population == 3

        # Test size mismatch
        with pytest.raises(ValueError):
            grid2.from_list([[1, 0], [0, 1]])  # Wrong size

    def test_get_bounding_box(self):
        """Test bounding box calculation."""
        grid = Grid(10, 10)

        # Empty grid
        assert grid.get_bounding_box() is None

        # Single cell
        grid.set_cell(5, 3, True)
        assert grid.get_bounding_box() == (5, 3, 5, 3)

        # Multiple cells
        grid.set_cell(2, 1, True)
        grid.set_cell(7, 8, True)
        bbox = grid.get_bounding_box()
        assert bbox == (2, 1, 7, 8)

    def test_equality(self):
        """Test grid equality comparison."""
        grid1 = Grid(3, 3)
        grid2 = Grid(3, 3)

        # Empty grids should be equal
        assert grid1 == grid2

        # Set same pattern in both
        grid1.set_cell(1, 1, True)
        grid2.set_cell(1, 1, True)
        assert grid1 == grid2

        # Different patterns
        grid2.set_cell(2, 2, True)
        assert grid1 != grid2

        # Different sizes
        grid3 = Grid(4, 4)
        assert grid1 != grid3

        # Different wrap settings
        grid4 = Grid(3, 3, wrap_edges=False)
        assert grid1 != grid4

        # Compare with non-grid
        assert grid1 != "not a grid"

    def test_string_representation(self):
        """Test string representation."""
        grid = Grid(3, 3)

        # Empty grid
        expected = "...\n...\n..."
        assert str(grid) == expected

        # Grid with some cells
        grid.set_cell(0, 0, True)
        grid.set_cell(1, 1, True)
        grid.set_cell(2, 2, True)

        expected = "*..\n.*.\n..*"
        assert str(grid) == expected
