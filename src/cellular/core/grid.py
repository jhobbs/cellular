"""Grid data structure for cellular automata."""

from typing import Tuple, Iterator, Optional
import numpy as np
import torch
import torch.nn.functional as F


class Grid:
    """Represents a 2D grid for cellular automata.

    The grid uses numpy arrays for efficient operations and supports
    both wraparound and bounded edge behavior.
    """

    def __init__(self, width: int, height: int, wrap_edges: bool = True) -> None:
        """Initialize a new grid.

        Args:
            width: Number of columns
            height: Number of rows
            wrap_edges: Whether edges wrap around (toroidal topology)
        """
        self.width = width
        self.height = height
        self.wrap_edges = wrap_edges
        self._cells = np.zeros((width, height), dtype=np.int8)
        self._previous_cells = np.zeros((width, height), dtype=np.int8)

        # PyTorch optimization: pre-allocate tensors and kernel
        # Set single-threaded to avoid conflicts with multiprocessing
        torch.set_num_threads(1)

        # PyTorch tensors for convolution (reused for efficiency)
        self._torch_input = torch.zeros(1, 1, height, width, dtype=torch.float32)
        self._torch_kernel = (
            torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

    @property
    def cells(self) -> np.ndarray:
        """Get the current cell array."""
        return self._cells

    @property
    def previous_cells(self) -> np.ndarray:
        """Get the previous cell array."""
        return self._previous_cells

    @property
    def shape(self) -> Tuple[int, int]:
        """Get grid dimensions as (width, height)."""
        return (self.width, self.height)

    def get_cell(self, x: int, y: int) -> bool:
        """Get the state of a cell.

        Args:
            x: Column coordinate
            y: Row coordinate

        Returns:
            True if cell is alive, False if dead

        Raises:
            IndexError: If coordinates are out of bounds and wrap_edges is False
        """
        if self.wrap_edges:
            x = x % self.width
            y = y % self.height
        elif not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")

        return bool(self._cells[x, y])

    def set_cell(self, x: int, y: int, alive: bool) -> None:
        """Set the state of a cell.

        Args:
            x: Column coordinate
            y: Row coordinate
            alive: Whether the cell should be alive

        Raises:
            IndexError: If coordinates are out of bounds and wrap_edges is False
        """
        if self.wrap_edges:
            x = x % self.width
            y = y % self.height
        elif not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")

        self._cells[x, y] = 1 if alive else 0

    def toggle_cell(self, x: int, y: int) -> bool:
        """Toggle the state of a cell.

        Args:
            x: Column coordinate
            y: Row coordinate

        Returns:
            New state of the cell
        """
        current_state = self.get_cell(x, y)
        new_state = not current_state
        self.set_cell(x, y, new_state)
        return new_state

    def clear(self) -> None:
        """Clear all cells (set all to dead)."""
        self._cells.fill(0)

    def randomize(self, probability: float = 0.1) -> None:
        """Randomly populate the grid.

        Args:
            probability: Chance each cell will be alive (0.0 to 1.0)
        """
        mask = np.random.random((self.width, self.height)) < probability
        self._cells[mask] = 1
        self._cells[~mask] = 0

    def copy_from(self, other: "Grid") -> None:
        """Copy cell states from another grid.

        Args:
            other: Source grid to copy from

        Raises:
            ValueError: If grids have different dimensions
        """
        if other.shape != self.shape:
            raise ValueError(f"Grid dimensions don't match: {other.shape} vs {self.shape}")

        self._cells[:] = other._cells

    def save_state(self) -> None:
        """Save current state to previous state."""
        self._previous_cells[:] = self._cells

    def get_changed_cells(self) -> Iterator[Tuple[int, int]]:
        """Get coordinates of cells that changed since last save_state().

        Yields:
            Tuples of (x, y) coordinates for changed cells
        """
        changed = self._cells != self._previous_cells
        coords = np.where(changed)
        for x, y in zip(coords[0], coords[1]):
            yield (int(x), int(y))

    @property
    def population(self) -> int:
        """Get the number of living cells."""
        return int(np.sum(self._cells > 0))

    def get_neighbors(self, x: int, y: int) -> int:
        """Count living neighbors of a cell.

        Args:
            x: Column coordinate
            y: Row coordinate

        Returns:
            Number of living neighbors (0-8)
        """
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if self.wrap_edges:
                    nx = nx % self.width
                    ny = ny % self.height
                    count += self._cells[nx, ny]
                elif 0 <= nx < self.width and 0 <= ny < self.height:
                    count += self._cells[nx, ny]

        return count

    def count_all_neighbors(self) -> np.ndarray:
        """Count neighbors for all cells using PyTorch-accelerated convolution.

        Returns:
            2D array with neighbor counts for each cell
        """
        # Use PyTorch for fast convolution
        # Grid uses (width, height) but PyTorch expects (height, width), so transpose
        self._torch_input[0, 0] = torch.from_numpy((self._cells.T > 0).astype(np.float32))

        if self.wrap_edges:
            # For toroidal topology, use circular padding
            padded = F.pad(self._torch_input, (1, 1, 1, 1), mode="circular")
            neighbors = F.conv2d(padded, self._torch_kernel)
        else:
            # For bounded topology, use zero padding
            neighbors = F.conv2d(self._torch_input, self._torch_kernel, padding=1)

        # Convert back to numpy and transpose back to (width, height)
        return neighbors[0, 0].numpy().astype(np.int8).T

    def to_list(self) -> list:
        """Convert grid to nested list for serialization.

        Returns:
            2D list representation of the grid
        """
        return self._cells.tolist()

    def from_list(self, data: list) -> None:
        """Load grid from nested list.

        Args:
            data: 2D list with cell states

        Raises:
            ValueError: If data dimensions don't match grid
        """
        arr = np.array(data, dtype=np.int8)
        if arr.shape != (self.width, self.height):
            raise ValueError(f"Data shape {arr.shape} doesn't match grid {self.shape}")

        self._cells[:] = arr

    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of living cells.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) or None if no living cells
        """
        living_coords = np.where(self._cells > 0)
        if len(living_coords[0]) == 0:
            return None

        min_x, max_x = int(living_coords[0].min()), int(living_coords[0].max())
        min_y, max_y = int(living_coords[1].min()), int(living_coords[1].max())

        return (min_x, min_y, max_x, max_y)

    def __eq__(self, other: object) -> bool:
        """Check if two grids are equal."""
        if not isinstance(other, Grid):
            return False
        return (
            self.shape == other.shape
            and self.wrap_edges == other.wrap_edges
            and np.array_equal(self._cells, other._cells)
        )

    def __str__(self) -> str:
        """String representation showing living cells as '*' and dead as '.'."""
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append("*" if self._cells[x, y] else ".")
            result.append("".join(row))
        return "\n".join(result)
