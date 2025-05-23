"""Frontend interfaces for cellular automata."""

from .tkinter_gui import TkinterGameOfLifeGUI
from .cli import CLIGameOfLife

__all__ = ["TkinterGameOfLifeGUI", "CLIGameOfLife"]
