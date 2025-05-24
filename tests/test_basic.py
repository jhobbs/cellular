"""Basic tests for cellular automata package."""

from cellular.core.grid import Grid
from cellular.core.game import GameOfLife
from cellular.core.patterns import PatternLibrary


def test_grid_creation():
    """Test basic grid creation and cell operations."""
    grid = Grid(10, 10)
    assert grid.width == 10
    assert grid.height == 10
    assert grid.get_cell(0, 0) is False

    grid.set_cell(5, 5, True)
    assert grid.get_cell(5, 5) is True


def test_game_creation():
    """Test basic game creation."""
    grid = Grid(5, 5)
    game = GameOfLife(grid)
    assert game.population == 0

    grid.set_cell(2, 2, True)
    assert game.population == 1


def test_pattern_library():
    """Test pattern library has some patterns."""
    library = PatternLibrary()
    patterns = library.list_patterns()
    assert len(patterns) > 0
    assert "Glider" in patterns


def test_blinker_pattern():
    """Test the blinker pattern oscillates correctly."""
    grid = Grid(5, 5, wrap_edges=False)
    game = GameOfLife(grid)

    # Create blinker pattern (vertical line)
    grid.set_cell(2, 1, True)
    grid.set_cell(2, 2, True)
    grid.set_cell(2, 3, True)

    initial_population = game.population
    assert initial_population == 3

    # Step once - should become horizontal
    game.step()
    assert game.population == 3
    assert grid.get_cell(1, 2) is True
    assert grid.get_cell(2, 2) is True
    assert grid.get_cell(3, 2) is True

    # Step again - should return to vertical
    game.step()
    assert game.population == 3
    assert grid.get_cell(2, 1) is True
    assert grid.get_cell(2, 2) is True
    assert grid.get_cell(2, 3) is True
