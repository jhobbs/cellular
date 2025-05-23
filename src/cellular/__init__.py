"""Cellular automata package with Conway's Game of Life implementation."""

__version__ = "0.1.0"

from .core.grid import Grid
from .core.game import GameOfLife
from .core.patterns import Pattern, PatternLibrary

__all__ = ["Grid", "GameOfLife", "Pattern", "PatternLibrary"]
