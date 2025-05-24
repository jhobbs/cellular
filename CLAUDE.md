# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a modular Python package for Conway's Game of Life with multiple frontend interfaces. The implementation focuses on performance optimization through vectorized operations, efficient rendering, and comprehensive pattern searching capabilities.

## Installation

### Basic Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package only (includes numpy, scipy, torch)
pip install -e .

# For CPU-only PyTorch (smaller download, no CUDA support):
pip install -e . --index-url https://download.pytorch.org/whl/cpu
```

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package + development tools (pytest, black, flake8, mypy, tox)
pip install -e ".[dev]"

# For CPU-only PyTorch + development tools:
pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu
```

## Running the Application

- **Tkinter GUI**: `cellular-tkinter` or `python -m cellular.frontends.tkinter_gui`
- **Test mode**: `cellular-tkinter --test` (runs for 3 seconds and auto-exits)
- **Command Line**: `cellular-cli` or `python -m cellular.frontends.cli`
- **CLI with pattern**: `cellular-cli --pattern Glider --toroidal --verbose`
- **CLI search mode**: `cellular-cli --search cycle_length:3 --search-attempts 500 --workers 4`
- **Sequential search**: `cellular-cli --search cycle_length:3 --workers 1` (forces single-threaded)
- **Save found patterns**: `cellular-cli --search cycle_length:6 --save-pattern 6peat_finishes`

## Testing and Quality Assurance

```bash
# Development testing (no test suite currently exists)
python -m cellular.frontends.cli --search cycle_length:3 --search-attempts 10 --verbose

# Linting and formatting
tox -e lint     # or flake8 src
tox -e type     # or mypy src  
tox -e format   # or black src

# All environments
tox
```

## Package Architecture

### Core Components (`src/cellular/core/`)

- **`grid.py`**: `Grid` class - 2D cellular grid with NumPy backend, supports wraparound edges
- **`game.py`**: `GameOfLife` class - Simulation engine implementing Conway's rules with cycle detection  
- **`patterns.py`**: `Pattern` and `PatternLibrary` classes - Pattern management and built-in pattern collection

### Frontend Interfaces (`src/cellular/frontends/`)

- **`tkinter_gui.py`**: Full-featured Tkinter GUI with pattern selection, statistics, and speed mode
- **`cli.py`**: Unified search implementation with both sequential and parallel processing capabilities

### Search Architecture (`cli.py`)

The CLI uses a **unified search system** where sequential searching is simply parallel searching with `workers=1`. Key components:

- **`parallel_search_for_condition()`**: Single search method handling both sequential (workers=1) and parallel (workers>1) cases
- **Worker batching**: Work is divided into batches for efficient load balancing across workers
- **Comprehensive statistics**: Tracks total attempts, generations computed, cycle distributions, and end states
- **Pattern saving**: Automatically saves successful configurations to `found_patterns/` directory
- **Interrupt handling**: Graceful Ctrl+C handling with partial result reporting

### Key Design Patterns

- **Separation of concerns**: Core logic independent of UI
- **Vectorized operations**: Uses NumPy/SciPy for performance
- **Unified search paths**: Sequential and parallel searches use identical code paths
- **Worker pool management**: Efficient multiprocessing with shared state and work stealing
- **Factory pattern**: Pattern library for creating known configurations

### Performance Optimizations

- **Vectorized neighbor counting**: `scipy.ndimage.convolve` for fast neighbor calculation
- **Batched parallel processing**: Work stealing algorithm prevents worker starvation
- **Statistics aggregation**: Efficient collection of end state statistics across workers
- **Memory management**: Bounded state history for cycle detection
- **Differential rendering**: Only redraws changed cells in GUI (tkinter frontend)

### Pattern Search Conditions

The CLI supports various search conditions:
- `cycle_length:N` - Finds patterns that stabilize with N-generation cycles
- `runs_for_at_least:N` - Patterns that survive for at least N generations
- `extinction_after:N` - Patterns that die out after exactly N generations  
- `population_threshold:N` - Patterns reaching population of N cells
- `stabilizes_with_population:N` - Patterns stabilizing at exactly N cells
- `bounding_box_size:WxH` - Patterns fitting within W×H bounding box

## Dependencies

- **Core**: `numpy>=1.20.0`, `scipy>=1.7.0`
- **GUI**: `tkinter` (built-in)
- **Development**: `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`, `tox`
- **Parallel processing**: `multiprocessing` (built-in)

## Search System Implementation Notes

- **No separate sequential method**: The old `search_for_condition()` method was removed to eliminate code duplication
- **Consistent return format**: All searches return `(found, attempts, result, end_state_stats)` tuple
- **Worker coordination**: Uses shared queues, locks, and flags for coordinating parallel workers
- **Progress monitoring**: Real-time progress updates with ETA calculations during long searches
- **Statistics tracking**: Comprehensive end state analysis including cycle detection and generation counting