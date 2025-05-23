# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a modular Python package for Conway's Game of Life with multiple frontend interfaces. The implementation focuses on performance optimization through vectorized operations, efficient rendering, and comprehensive testing.

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Running the Application

- **Tkinter GUI**: `cellular-tkinter` or `python -m cellular.frontends.tkinter_gui`
- **Test mode**: `cellular-tkinter --test` (runs for 3 seconds and auto-exits)

## Testing and Quality Assurance

- **Run all tests**: `pytest` or `tox`
- **Test with coverage**: `pytest --cov=src/cellular`
- **Run specific tests**: `pytest tests/core/test_game.py`
- **Linting**: `tox -e lint` or `flake8 src tests`
- **Type checking**: `tox -e type` or `mypy src`
- **Code formatting**: `tox -e format` or `black src tests`
- **All environments**: `tox` (tests Python 3.8-3.12)

## Package Architecture

### Core Components (`src/cellular/core/`)

- **`grid.py`**: `Grid` class - 2D cellular grid with NumPy backend, supports wraparound edges
- **`game.py`**: `GameOfLife` class - Simulation engine implementing Conway's rules with cycle detection  
- **`patterns.py`**: `Pattern` and `PatternLibrary` classes - Pattern management and built-in pattern collection

### Frontend Interfaces (`src/cellular/frontends/`)

- **`tkinter_gui.py`**: Full-featured Tkinter GUI with pattern selection, statistics, and speed mode
- Future frontends can be added here (web, CLI, etc.)

### Key Design Patterns

- **Separation of concerns**: Core logic independent of UI
- **Vectorized operations**: Uses NumPy/SciPy for performance
- **Observer pattern**: Game state changes trigger UI updates  
- **Strategy pattern**: Multiple frontend implementations
- **Factory pattern**: Pattern library for creating known configurations

### Performance Optimizations

- **Vectorized neighbor counting**: `scipy.ndimage.convolve` for fast neighbor calculation
- **Differential rendering**: Only redraws changed cells in GUI
- **Canvas object caching**: Reuses Tkinter objects instead of recreating
- **Speed mode**: Batch processing of generations without per-frame rendering
- **Memory management**: Bounded state history for cycle detection

### Testing Structure

- **Unit tests**: Comprehensive coverage of core logic (`tests/core/`)
- **Integration tests**: Frontend testing with mocked dependencies (`tests/frontends/`)
- **Property-based testing**: Uses pytest fixtures for consistent test setup
- **Performance tests**: Can be added for optimization validation

## Dependencies

- **Core**: `numpy>=1.20.0`, `scipy>=1.7.0`
- **GUI**: `tkinter` (built-in)
- **Development**: `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`, `tox`

## Common Development Tasks

- **Add new pattern**: Add to `PatternLibrary._load_builtin_patterns()`
- **Add new frontend**: Create in `src/cellular/frontends/` following `tkinter_gui.py` pattern
- **Extend game rules**: Modify `GameOfLife._apply_rules()` method
- **Add performance metrics**: Extend `GameOfLife.get_statistics()`