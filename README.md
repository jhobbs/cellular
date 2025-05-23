# Cellular Automata

A high-performance implementation of Conway's Game of Life with multiple frontend interfaces.

## Features

- Optimized core engine using NumPy vectorization
- Multiple frontend interfaces (Tkinter GUI included)
- Pattern save/load functionality
- Cycle detection
- Speed mode for rapid simulation
- Comprehensive test suite

## Installation

### Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd cellular

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Using tox

```bash
# Run tests across all Python versions
tox

# Run specific environment
tox -e py311

# Run linting
tox -e lint

# Run type checking
tox -e type

# Format code
tox -e format
```

## Usage

### Tkinter GUI

```bash
cellular-tkinter
```

### Command Line Interface

```bash
# Run random simulation
cellular-cli --width 50 --height 50 --population 0.1

# Run specific pattern
cellular-cli --pattern Glider --toroidal --verbose

# Run R-pentomino with detailed output
cellular-cli --pattern "R-pentomino" --width 100 --height 100 --show-grid

# List available patterns
cellular-cli --list-patterns

# Get help
cellular-cli --help

# Search modes - find configurations meeting specific conditions
cellular-cli --search cycle_length:2 --search-attempts 100
cellular-cli --search runs_for_at_least:500 --population 0.15
cellular-cli --search extinction_after:50 --width 20 --height 20
cellular-cli --search population_threshold:100 --toroidal
cellular-cli --search stabilizes_with_population:25
cellular-cli --search bounding_box_size:15x10
```

### Programmatic Usage

```python
from cellular.core.game import GameOfLife
from cellular.core.grid import Grid

# Create a 50x50 grid
grid = Grid(50, 50)
game = GameOfLife(grid)

# Set some initial pattern
grid.set_cell(25, 25, True)
grid.set_cell(25, 26, True)
grid.set_cell(25, 27, True)

# Run simulation
for _ in range(100):
    game.step()
    print(f"Generation {game.generation}: {game.population} cells alive")
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cellular

# Run specific test file
pytest tests/core/test_game.py
```

## Development

The project uses:
- **tox** for testing across Python versions
- **pytest** for unit testing
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

## Architecture

- `src/cellular/core/` - Core game logic and data structures
- `src/cellular/frontends/` - User interface implementations
- `tests/` - Test suite