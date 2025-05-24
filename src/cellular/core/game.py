"""Conway's Game of Life implementation."""

from typing import Deque, Dict, Tuple
from collections import deque
import numpy as np

from .grid import Grid


class GameOfLife:
    """Conway's Game of Life simulation engine.

    Implements the classic rules:
    - Live cell with 2-3 neighbors survives
    - Dead cell with exactly 3 neighbors becomes alive
    - All other cells die or stay dead
    """

    def __init__(self, grid: Grid) -> None:
        """Initialize the game with a grid.

        Args:
            grid: The cellular grid to simulate
        """
        self.grid = grid
        self._generation = 0
        self._population_history: Deque[int] = deque(maxlen=100)
        self._state_history: Deque[bytes] = deque(maxlen=1000)
        self._seen_states: Dict[bytes, int] = {}
        self._cycle_detected = False
        self._cycle_length = 0
        self._cycle_start_generation = 0

        # Track initial population
        self._update_population_history()

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    @property
    def population(self) -> int:
        """Current number of living cells."""
        return self.grid.population

    @property
    def population_history(self) -> list:
        """History of population counts."""
        return list(self._population_history)

    @property
    def cycle_detected(self) -> bool:
        """Whether a cycle has been detected."""
        return self._cycle_detected

    @property
    def cycle_length(self) -> int:
        """Length of detected cycle (0 if no cycle)."""
        return self._cycle_length

    @property
    def cycle_start_generation(self) -> int:
        """Generation where cycle started (0 if no cycle)."""
        return self._cycle_start_generation

    def step(self) -> None:
        """Advance the simulation by one generation."""
        # Save current state before updating
        self.grid.save_state()

        # Check for cycles before updating
        self._check_for_cycles()

        # Apply Conway's rules
        self._apply_rules()

        # Update tracking
        self._generation += 1
        self._update_population_history()

    def _apply_rules(self) -> None:
        """Apply Conway's Game of Life rules to update the grid."""
        # Count neighbors for all cells efficiently
        neighbor_counts = self.grid.count_all_neighbors()

        # Apply rules vectorially
        cells = self.grid.cells

        # Birth: dead cell with exactly 3 neighbors
        birth_mask = (cells == 0) & (neighbor_counts == 3)

        # Death: live cell with < 2 or > 3 neighbors
        death_mask = (cells > 0) & ((neighbor_counts < 2) | (neighbor_counts > 3))

        # Apply changes
        cells[death_mask] = 0
        cells[birth_mask] = 1

    def _update_population_history(self) -> None:
        """Update the population history."""
        self._population_history.append(self.population)

    def _check_for_cycles(self) -> None:
        """Check if the current state has been seen before (cycle detection)."""
        if self._cycle_detected:
            return

        # Convert current state to bytes for hashing
        current_state = self.grid.cells.tobytes()

        # Check if we've seen this state
        if current_state in self._seen_states:
            # Cycle detected!
            first_occurrence = self._seen_states[current_state]
            self._cycle_detected = True
            self._cycle_length = self._generation - first_occurrence
            self._cycle_start_generation = first_occurrence
            return

        # Record this state
        self._seen_states[current_state] = self._generation
        self._state_history.append(current_state)

        # Clean up old states to prevent memory growth
        if len(self._state_history) > 900:
            old_state = self._state_history[0]
            if old_state in self._seen_states:
                # Only delete if it's still the first occurrence
                if self._seen_states[old_state] == self._generation - len(self._state_history) + 1:
                    del self._seen_states[old_state]

    def reset(self, clear_grid: bool = True) -> None:
        """Reset the simulation.

        Args:
            clear_grid: Whether to clear the grid as well
        """
        if clear_grid:
            self.grid.clear()

        self._generation = 0
        self._population_history.clear()
        self._state_history.clear()
        self._seen_states.clear()
        self._cycle_detected = False
        self._cycle_length = 0
        self._cycle_start_generation = 0

        self._update_population_history()

    def clear_cycle_detection(self) -> None:
        """Clear cycle detection state while preserving generation and population history.

        This should be called when the grid is manually modified to reset cycle detection
        since the state space has changed.
        """
        self._cycle_detected = False
        self._cycle_length = 0
        self._cycle_start_generation = 0
        self._seen_states.clear()
        self._state_history.clear()

    def run_until_stable(self, max_generations: int = 10000) -> Tuple[int, str]:
        """Run simulation until it becomes stable or cycles.

        Args:
            max_generations: Maximum generations to run

        Returns:
            Tuple of (final_generation, reason) where reason is one of:
            'cycle', 'extinction', 'max_generations'
        """
        # start_generation = self._generation  # noqa: F841

        for _ in range(max_generations):
            self.step()

            if self._cycle_detected:
                return self._generation, "cycle"

            if self.population == 0:
                return self._generation, "extinction"

        return self._generation, "max_generations"

    def get_population_change_rate(self, window_size: int = 10) -> float:
        """Calculate recent population change rate.

        Args:
            window_size: Number of recent generations to consider

        Returns:
            Average population change per generation
        """
        if len(self._population_history) < 2:
            return 0.0

        recent_history = list(self._population_history)[-window_size:]
        if len(recent_history) < 2:
            return 0.0

        changes = np.diff(recent_history)
        return float(np.mean(changes))

    def save_state(self) -> Dict:
        """Save complete game state for serialization.

        Returns:
            Dictionary containing all game state
        """
        return {
            "generation": self._generation,
            "grid_data": self.grid.to_list(),
            "grid_width": self.grid.width,
            "grid_height": self.grid.height,
            "wrap_edges": self.grid.wrap_edges,
            "population_history": list(self._population_history),
            "cycle_detected": self._cycle_detected,
            "cycle_length": self._cycle_length,
            "cycle_start_generation": self._cycle_start_generation,
        }

    def load_state(self, state: Dict) -> None:
        """Load complete game state from serialization.

        Args:
            state: Dictionary containing game state

        Raises:
            ValueError: If state is incompatible with current grid
        """
        # Validate grid compatibility
        if state["grid_width"] != self.grid.width or state["grid_height"] != self.grid.height:
            raise ValueError(
                f"Grid size mismatch: saved {state['grid_width']}x"
                f"{state['grid_height']} vs current {self.grid.width}x"
                f"{self.grid.height}"
            )

        # Load grid data
        self.grid.from_list(state["grid_data"])

        # Load game state
        self._generation = state["generation"]
        self._population_history = deque(state["population_history"], maxlen=100)
        self._cycle_detected = state["cycle_detected"]
        self._cycle_length = state["cycle_length"]
        self._cycle_start_generation = state["cycle_start_generation"]

        # Clear state history (it's not serialized)
        self._state_history.clear()
        self._seen_states.clear()

    def get_statistics(self) -> Dict:
        """Get comprehensive simulation statistics.

        Returns:
            Dictionary with various statistics
        """
        bbox = self.grid.get_bounding_box()

        stats = {
            "generation": self._generation,
            "population": self.population,
            "population_change_rate": self.get_population_change_rate(),
            "population_history": list(self._population_history),
            "cycle_detected": self._cycle_detected,
            "cycle_length": self._cycle_length,
            "cycle_start_generation": self._cycle_start_generation,
            "grid_size": self.grid.shape,
            "population_density": self.population / (self.grid.width * self.grid.height),
        }

        if bbox:
            stats["bounding_box"] = bbox
            box_width = bbox[2] - bbox[0] + 1
            box_height = bbox[3] - bbox[1] + 1
            stats["bounding_box_size"] = (box_width, box_height)
            stats["bounding_box_area"] = box_width * box_height
        else:
            stats["bounding_box"] = None
            stats["bounding_box_size"] = (0, 0)
            stats["bounding_box_area"] = 0

        return stats
