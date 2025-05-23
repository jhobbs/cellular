"""Tests for the GameOfLife class."""

import pytest
from cellular.core.grid import Grid
from cellular.core.game import GameOfLife


class TestGameOfLife:
    """Test cases for the GameOfLife class."""

    def test_initialization(self):
        """Test game initialization."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        assert game.grid is grid
        assert game.generation == 0
        assert game.population == 0
        assert len(game.population_history) == 1
        assert not game.cycle_detected
        assert game.cycle_length == 0
        assert game.cycle_start_generation == 0

    def test_still_life_block(self):
        """Test that a block pattern is stable (still life)."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Create a 2x2 block
        grid.set_cell(4, 4, True)
        grid.set_cell(4, 5, True)
        grid.set_cell(5, 4, True)
        grid.set_cell(5, 5, True)

        initial_population = game.population
        assert initial_population == 4

        # Run several generations
        for _ in range(5):
            game.step()

        # Should remain unchanged
        assert game.population == initial_population
        assert grid.get_cell(4, 4)
        assert grid.get_cell(4, 5)
        assert grid.get_cell(5, 4)
        assert grid.get_cell(5, 5)
        assert game.generation == 5

    def test_oscillator_blinker(self):
        """Test blinker oscillator (period 2)."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Create vertical blinker
        grid.set_cell(5, 4, True)
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)

        assert game.population == 3

        # After one step, should be horizontal
        game.step()
        assert game.population == 3
        assert grid.get_cell(4, 5)
        assert grid.get_cell(5, 5)
        assert grid.get_cell(6, 5)
        assert not grid.get_cell(5, 4)
        assert not grid.get_cell(5, 6)

        # After another step, should be vertical again
        game.step()
        assert game.population == 3
        assert grid.get_cell(5, 4)
        assert grid.get_cell(5, 5)
        assert grid.get_cell(5, 6)
        assert not grid.get_cell(4, 5)
        assert not grid.get_cell(6, 5)

    def test_extinction(self):
        """Test pattern that goes extinct."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Single cell (will die)
        grid.set_cell(5, 5, True)
        assert game.population == 1

        # After one step, should be dead
        game.step()
        assert game.population == 0
        assert game.generation == 1

    def test_population_history(self):
        """Test population history tracking."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Start with some cells
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)
        grid.set_cell(6, 5, True)

        # Update population history after setting cells
        game._update_population_history()

        initial_pop = game.population
        history = game.population_history
        assert len(history) == 2  # Initial 0 + updated population
        assert history[-1] == initial_pop

        # Run a few steps
        for i in range(3):
            game.step()
            new_history = game.population_history
            assert len(new_history) == i + 3  # Adjusted for the extra entry
            assert new_history[-1] == game.population

    def test_cycle_detection(self):
        """Test cycle detection with blinker."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Create blinker (period 2)
        grid.set_cell(5, 4, True)
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)

        # Run until cycle is detected
        for _ in range(10):
            if game.cycle_detected:
                break
            game.step()

        assert game.cycle_detected
        assert game.cycle_length == 2
        assert game.cycle_start_generation <= 2

    def test_reset(self):
        """Test game reset functionality."""
        grid = Grid(5, 5)
        game = GameOfLife(grid)

        # Set up initial state
        grid.set_cell(2, 2, True)
        game.step()
        game.step()

        assert game.generation == 2
        assert len(game.population_history) > 1

        # Reset without clearing grid
        game.reset(clear_grid=False)
        assert game.generation == 0
        assert len(game.population_history) == 1
        assert not game.cycle_detected
        # Grid should not be cleared, but the cell may have died during simulation

        # Reset with clearing grid
        game.reset(clear_grid=True)
        assert game.generation == 0
        assert game.population == 0
        assert not grid.get_cell(2, 2)  # Grid cleared

    def test_run_until_stable_extinction(self):
        """Test run_until_stable with extinction."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Single cell that will die
        grid.set_cell(5, 5, True)

        final_gen, reason = game.run_until_stable(max_generations=100)

        assert reason == "extinction"
        assert final_gen == 1
        assert game.population == 0

    def test_run_until_stable_cycle(self):
        """Test run_until_stable with cycle detection."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Blinker pattern
        grid.set_cell(5, 4, True)
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)

        final_gen, reason = game.run_until_stable(max_generations=100)

        assert reason == "cycle"
        assert game.cycle_detected
        assert game.cycle_length == 2

    def test_run_until_stable_max_generations(self):
        """Test run_until_stable hitting max generations."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # R-pentomino (long-lived pattern)
        grid.set_cell(5, 4, True)
        grid.set_cell(6, 4, True)
        grid.set_cell(4, 5, True)
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)

        max_gens = 10
        final_gen, reason = game.run_until_stable(max_generations=max_gens)

        assert reason == "max_generations"
        assert final_gen == max_gens

    def test_population_change_rate(self):
        """Test population change rate calculation."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Start with stable population (2x2 block)
        grid.set_cell(4, 4, True)
        grid.set_cell(4, 5, True)
        grid.set_cell(5, 4, True)
        grid.set_cell(5, 5, True)  # 2x2 block

        # Update population history to capture initial state
        game._update_population_history()

        # Run a few generations (block should be stable)
        for _ in range(5):
            game.step()

        # Rate should be 0 for stable pattern (excluding the initial jump)
        # Use a window that excludes the 0->4 transition
        rate = game.get_population_change_rate(window_size=5)
        assert rate == 0.0

        # Test with limited history
        rate_small_window = game.get_population_change_rate(window_size=2)
        assert rate_small_window == 0.0

    def test_save_and_load_state(self):
        """Test state saving and loading."""
        grid = Grid(5, 5)
        game = GameOfLife(grid)

        # Set up initial state
        grid.set_cell(2, 2, True)
        grid.set_cell(2, 3, True)
        game.step()
        game.step()

        # Save state
        saved_state = game.save_state()

        # Verify state contents
        assert saved_state["generation"] == 2
        assert saved_state["grid_width"] == 5
        assert saved_state["grid_height"] == 5
        assert saved_state["wrap_edges"] is True
        assert "grid_data" in saved_state
        assert "population_history" in saved_state

        # Modify game state
        game.step()
        game.step()
        assert game.generation == 4

        # Load saved state
        game.load_state(saved_state)

        # Verify restoration
        assert game.generation == 2
        assert len(game.population_history) == len(saved_state["population_history"])

    def test_load_state_size_mismatch(self):
        """Test loading state with mismatched grid size."""
        grid = Grid(5, 5)
        game = GameOfLife(grid)

        # Create state with different grid size
        wrong_state = {
            "generation": 1,
            "grid_width": 3,
            "grid_height": 3,
            "wrap_edges": True,
            "grid_data": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "population_history": [1],
            "cycle_detected": False,
            "cycle_length": 0,
            "cycle_start_generation": 0,
        }

        with pytest.raises(ValueError, match="Grid size mismatch"):
            game.load_state(wrong_state)

    def test_get_statistics(self):
        """Test statistics gathering."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Set up pattern
        grid.set_cell(4, 4, True)
        grid.set_cell(4, 5, True)
        grid.set_cell(5, 4, True)
        grid.set_cell(5, 5, True)

        game.step()
        stats = game.get_statistics()

        # Check required fields
        assert "generation" in stats
        assert "population" in stats
        assert "population_change_rate" in stats
        assert "cycle_detected" in stats
        assert "cycle_length" in stats
        assert "cycle_start_generation" in stats
        assert "grid_size" in stats
        assert "population_density" in stats
        assert "bounding_box" in stats
        assert "bounding_box_size" in stats
        assert "bounding_box_area" in stats

        # Check values
        assert stats["generation"] == 1
        assert stats["population"] == 4
        assert stats["grid_size"] == (10, 10)
        assert stats["population_density"] == 4.0 / 100
        assert stats["bounding_box"] == (4, 4, 5, 5)
        assert stats["bounding_box_size"] == (2, 2)
        assert stats["bounding_box_area"] == 4

    def test_get_statistics_empty_grid(self):
        """Test statistics with empty grid."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        stats = game.get_statistics()

        assert stats["population"] == 0
        assert stats["population_density"] == 0.0
        assert stats["bounding_box"] is None
        assert stats["bounding_box_size"] == (0, 0)
        assert stats["bounding_box_area"] == 0

    def test_conway_rules_underpopulation(self):
        """Test Conway's rule: underpopulation (< 2 neighbors)."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Single cell (0 neighbors) - should die
        grid.set_cell(5, 5, True)
        game.step()
        assert not grid.get_cell(5, 5)

        # Two cells next to each other (1 neighbor each) - should die
        grid.clear()
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)
        game.step()
        assert not grid.get_cell(5, 5)
        assert not grid.get_cell(5, 6)

    def test_conway_rules_survival(self):
        """Test Conway's rule: survival (2-3 neighbors)."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Create L-shape with cells having 2-3 neighbors
        grid.set_cell(5, 5, True)  # Will have 2 neighbors
        grid.set_cell(5, 6, True)  # Will have 2 neighbors
        grid.set_cell(6, 5, True)  # Will have 2 neighbors

        game.step()

        # All should survive (each has exactly 2 neighbors)
        # This will actually become a 2x2 block due to birth rule
        assert game.population == 4  # Block pattern

    def test_conway_rules_overpopulation(self):
        """Test Conway's rule: overpopulation (> 3 neighbors)."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Create a 3x3 filled square - center cell will have 8 neighbors
        # (overpopulation)
        for x in range(4, 7):
            for y in range(4, 7):
                grid.set_cell(x, y, True)

        initial_population = game.population
        assert initial_population == 9

        game.step()

        # Many cells should die from overpopulation, some may be born
        # The center cell (5,5) definitely should die (8 neighbors > 3)
        assert not grid.get_cell(5, 5)  # Center should be dead
        assert (
            game.population < initial_population
        )  # Overall population should decrease

    def test_conway_rules_birth(self):
        """Test Conway's rule: birth (exactly 3 neighbors)."""
        grid = Grid(10, 10)
        game = GameOfLife(grid)

        # Create L-shape where empty corner has exactly 3 neighbors
        grid.set_cell(5, 5, True)
        grid.set_cell(5, 6, True)
        grid.set_cell(6, 5, True)
        # Position (6, 6) has exactly 3 neighbors and should be born

        game.step()

        # Should form a 2x2 block
        assert grid.get_cell(6, 6)  # New cell born
        assert game.population == 4
