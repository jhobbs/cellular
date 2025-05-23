"""Common Conway's Game of Life patterns and pattern management."""

from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

from .grid import Grid


class Pattern:
    """Represents a Game of Life pattern."""

    def __init__(
        self,
        name: str,
        cells: List[Tuple[int, int]],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a pattern.

        Args:
            name: Pattern name
            cells: List of (x, y) coordinates for living cells
            description: Optional description
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.cells = cells
        self.description = description
        self.metadata = metadata or {}

    def apply_to_grid(self, grid: Grid, offset_x: int = 0, offset_y: int = 0) -> None:
        """Apply this pattern to a grid.

        Args:
            grid: Target grid
            offset_x: Horizontal offset
            offset_y: Vertical offset
        """
        grid.clear()
        for x, y in self.cells:
            try:
                grid.set_cell(x + offset_x, y + offset_y, True)
            except IndexError:
                # Skip cells that fall outside grid bounds
                pass

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box of the pattern.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if not self.cells:
            return (0, 0, 0, 0)

        xs, ys = zip(*self.cells)
        return (min(xs), min(ys), max(xs), max(ys))

    def get_size(self) -> Tuple[int, int]:
        """Get pattern size.

        Returns:
            Tuple of (width, height)
        """
        min_x, min_y, max_x, max_y = self.get_bounding_box()
        return (max_x - min_x + 1, max_y - min_y + 1)

    def normalize(self) -> "Pattern":
        """Return a new pattern with coordinates normalized to start at (0, 0).

        Returns:
            New Pattern instance with normalized coordinates
        """
        if not self.cells:
            return Pattern(self.name, [], self.description, self.metadata.copy())

        min_x, min_y, _, _ = self.get_bounding_box()
        normalized_cells = [(x - min_x, y - min_y) for x, y in self.cells]

        return Pattern(
            self.name, normalized_cells, self.description, self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "cells": self.cells,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create pattern from dictionary.

        Args:
            data: Dictionary with pattern data

        Returns:
            New Pattern instance
        """
        # Convert cells from list of lists to list of tuples
        cells = [tuple(cell) for cell in data["cells"]]

        return cls(
            name=data["name"],
            cells=cells,
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_grid(cls, grid: Grid, name: str, description: str = "") -> "Pattern":
        """Create pattern from current grid state.

        Args:
            grid: Source grid
            name: Pattern name
            description: Optional description

        Returns:
            New Pattern instance
        """
        cells = []
        for x in range(grid.width):
            for y in range(grid.height):
                if grid.get_cell(x, y):
                    cells.append((x, y))

        metadata = {"source_grid_size": grid.shape, "population": len(cells)}

        return cls(name, cells, description, metadata)


class PatternLibrary:
    """Manages a collection of patterns."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        """Initialize pattern library.

        Args:
            storage_dir: Directory for storing patterns (defaults to 'patterns')
        """
        self.storage_dir = Path(storage_dir or "patterns")
        self.storage_dir.mkdir(exist_ok=True)
        self._patterns: Dict[str, Pattern] = {}
        self._load_builtin_patterns()

    def _load_builtin_patterns(self) -> None:
        """Load built-in common patterns."""
        # Still life patterns
        self.add_pattern(
            Pattern("Block", [(0, 0), (0, 1), (1, 0), (1, 1)], "2x2 still life block")
        )

        self.add_pattern(
            Pattern(
                "Beehive",
                [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)],
                "Beehive still life",
            )
        )

        self.add_pattern(
            Pattern(
                "Loaf",
                [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (3, 2), (2, 3)],
                "Loaf still life",
            )
        )

        # Oscillators
        self.add_pattern(
            Pattern("Blinker", [(0, 1), (1, 1), (2, 1)], "Period-2 oscillator")
        )

        self.add_pattern(
            Pattern(
                "Toad",
                [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)],
                "Period-2 oscillator",
            )
        )

        self.add_pattern(
            Pattern(
                "Beacon",
                [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)],
                "Period-2 oscillator",
            )
        )

        self.add_pattern(
            Pattern(
                "Pulsar",
                [
                    # Top part
                    (2, 0),
                    (3, 0),
                    (4, 0),
                    (8, 0),
                    (9, 0),
                    (10, 0),
                    (0, 2),
                    (5, 2),
                    (7, 2),
                    (12, 2),
                    (0, 3),
                    (5, 3),
                    (7, 3),
                    (12, 3),
                    (0, 4),
                    (5, 4),
                    (7, 4),
                    (12, 4),
                    (2, 5),
                    (3, 5),
                    (4, 5),
                    (8, 5),
                    (9, 5),
                    (10, 5),
                    # Bottom part (mirrored)
                    (2, 7),
                    (3, 7),
                    (4, 7),
                    (8, 7),
                    (9, 7),
                    (10, 7),
                    (0, 8),
                    (5, 8),
                    (7, 8),
                    (12, 8),
                    (0, 9),
                    (5, 9),
                    (7, 9),
                    (12, 9),
                    (0, 10),
                    (5, 10),
                    (7, 10),
                    (12, 10),
                    (2, 12),
                    (3, 12),
                    (4, 12),
                    (8, 12),
                    (9, 12),
                    (10, 12),
                ],
                "Period-3 oscillator",
            )
        )

        # Spaceships
        self.add_pattern(
            Pattern(
                "Glider",
                [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
                "Smallest spaceship, period-4",
            )
        )

        self.add_pattern(
            Pattern(
                "Lightweight Spaceship",
                [
                    (0, 0),
                    (3, 0),
                    (4, 1),
                    (0, 2),
                    (4, 2),
                    (1, 3),
                    (2, 3),
                    (3, 3),
                    (4, 3),
                ],
                "LWSS - Period-4 spaceship",
            )
        )

        # Methuselahs
        self.add_pattern(
            Pattern(
                "R-pentomino",
                [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
                "Famous methuselah that stabilizes after 1103 generations",
            )
        )

        self.add_pattern(
            Pattern(
                "Diehard",
                [(6, 0), (0, 1), (1, 1), (1, 2), (5, 2), (6, 2), (7, 2)],
                "Dies after exactly 130 generations",
            )
        )

        self.add_pattern(
            Pattern(
                "Acorn",
                [(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)],
                "Takes 5206 generations to stabilize",
            )
        )

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the library.

        Args:
            pattern: Pattern to add
        """
        self._patterns[pattern.name] = pattern

    def get_pattern(self, name: str) -> Optional[Pattern]:
        """Get a pattern by name.

        Args:
            name: Pattern name

        Returns:
            Pattern instance or None if not found
        """
        return self._patterns.get(name)

    def list_patterns(self) -> List[str]:
        """Get list of all pattern names.

        Returns:
            List of pattern names
        """
        return list(self._patterns.keys())

    def get_patterns_by_category(self) -> Dict[str, List[str]]:
        """Get patterns organized by category.

        Returns:
            Dictionary mapping categories to pattern name lists
        """
        categories = {
            "Still Life": ["Block", "Beehive", "Loaf"],
            "Oscillators": ["Blinker", "Toad", "Beacon", "Pulsar"],
            "Spaceships": ["Glider", "Lightweight Spaceship"],
            "Methuselahs": ["R-pentomino", "Diehard", "Acorn"],
            "Custom": [],
        }

        # Add any custom patterns to the Custom category
        all_builtin = set()
        for cat_patterns in categories.values():
            all_builtin.update(cat_patterns)

        for name in self._patterns:
            if name not in all_builtin:
                categories["Custom"].append(name)

        # Remove empty categories
        return {cat: patterns for cat, patterns in categories.items() if patterns}

    def save_pattern(self, pattern: Pattern, filename: Optional[str] = None) -> None:
        """Save a pattern to disk.

        Args:
            pattern: Pattern to save
            filename: Optional filename (defaults to pattern name)
        """
        if filename is None:
            filename = f"{pattern.name.replace(' ', '_').lower()}.json"

        filepath = self.storage_dir / filename
        with open(filepath, "w") as f:
            json.dump(pattern.to_dict(), f, indent=2)

    def load_pattern(self, filename: str) -> Pattern:
        """Load a pattern from disk.

        Args:
            filename: Filename to load from

        Returns:
            Loaded Pattern instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = self.storage_dir / filename

        with open(filepath, "r") as f:
            data = json.load(f)

        pattern = Pattern.from_dict(data)
        self.add_pattern(pattern)
        return pattern

    def load_all_patterns(self) -> None:
        """Load all patterns from the storage directory."""
        for filepath in self.storage_dir.glob("*.json"):
            try:
                self.load_pattern(filepath.name)
            except (ValueError, KeyError) as e:
                print(f"Warning: Failed to load pattern from {filepath.name}: {e}")

    def save_grid_as_pattern(
        self,
        grid: Grid,
        name: str,
        description: str = "",
        filename: Optional[str] = None,
    ) -> Pattern:
        """Save current grid state as a new pattern.

        Args:
            grid: Source grid
            name: Pattern name
            description: Optional description
            filename: Optional filename for saving

        Returns:
            Created Pattern instance
        """
        pattern = Pattern.from_grid(grid, name, description)
        self.add_pattern(pattern)

        if filename is not None:
            self.save_pattern(pattern, filename)

        return pattern
