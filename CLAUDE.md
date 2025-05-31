# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Tool Configuration

The following tools should be enabled for this project:
- Task
- Bash
- Glob
- Grep
- LS
- Read
- Edit
- MultiEdit
- Write
- NotebookRead
- NotebookEdit
- WebFetch
- TodoRead
- TodoWrite
- WebSearch

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

### Tkinter GUI
- **Start GUI**: `cellular-tkinter` or `python -m cellular.frontends.tkinter_gui`
- **Test mode**: `cellular-tkinter --test` (runs for 3 seconds and auto-exits)
- **Pattern loading**: File dialog defaults to `found_patterns/` directory if it exists
- **Pattern reset**: Reset button reloads the last loaded pattern (shows as "Reset Pattern")

### Command Line Interface
- **Basic run**: `cellular-cli` or `python -m cellular.frontends.cli`
- **With pattern**: `cellular-cli --pattern Glider --toroidal --verbose`
- **List patterns**: `cellular-cli --list-patterns`

### Pattern Search Modes
- **Simple search**: `cellular-cli --search cycle_length:3 --search-attempts 500 --workers 4`
- **Sequential search**: `cellular-cli --search cycle_length:3 --workers 1` (forces single-threaded)
- **Save patterns**: `cellular-cli --search cycle_length:6 --save-pattern 6peat_finishes`
- **Composite search**: `cellular-cli --search composite:1,2,3;true;false` (exclude cycles 1,2,3; allow extinction; no max gen)
  - Format: `composite:exclude_cycles;allow_extinction;allow_max_generations`
  - Example: `composite:2,3,4;true;true` finds patterns with cycles (except 2,3,4), extinctions, or max generations
  - Saved files include termination reason: `_cycle5`, `_extinct`, `_maxgen`

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
- **Real-time pattern saving**: Separate `pattern_saver` thread monitors queue and saves patterns immediately upon discovery
- **Comprehensive statistics**: Tracks total attempts, generations computed, cycle distributions, and end states
- **Pattern saving**: Automatically saves successful configurations to `found_patterns/` directory with reason-based filenames
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
- `composite:exclude_cycles;allow_extinction;allow_max_gens` - Flexible multi-condition search
  - `exclude_cycles`: Comma-separated list of cycle lengths to exclude (e.g., `1,2,3`)
  - `allow_extinction`: `true` to include extinct patterns, `false` to exclude
  - `allow_max_gens`: `true` to include patterns hitting generation limit, `false` to exclude

## Dependencies

- **Core**: `numpy>=1.20.0`, `scipy>=1.7.0`, `torch>=2.0.0,<2.8.0`
- **GUI**: `tkinter` (built-in)
- **Development**: `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`, `tox`
- **Parallel processing**: `multiprocessing` (built-in), `threading` for real-time pattern saving

## Search System Implementation Notes

- **No separate sequential method**: The old `search_for_condition()` method was removed to eliminate code duplication
- **Consistent return format**: All searches return `(found, attempts, result, end_state_stats)` tuple
- **Worker coordination**: Uses shared queues, locks, and flags for coordinating parallel workers
- **Real-time saving**: Pattern saver thread immediately saves patterns as they're discovered, not at run completion
- **Progress monitoring**: Real-time progress updates with ETA calculations during long searches
- **Statistics tracking**: Comprehensive end state analysis including cycle detection and generation counting
- **Filename conventions**: Saved patterns include termination reason in filename (e.g., `pattern_cycle3_timestamp.json`)

## Recent Feature Additions

### CLI Enhancements
- **Composite search conditions**: Search for patterns matching multiple criteria simultaneously
- **Real-time pattern saving**: Patterns save immediately upon discovery via dedicated thread
- **Reason-based filenames**: Pattern files include termination reason (`_cycle5`, `_extinct`, `_maxgen`)
- **Console output**: Prints save location and details when patterns are saved

### GUI Improvements
- **Smart file dialog**: Defaults to `found_patterns/` directory when loading patterns
- **Pattern reload**: Reset button remembers and reloads the last loaded pattern
- **Visual feedback**: Window title shows loaded pattern name, reset button shows "Reset Pattern" when pattern loaded