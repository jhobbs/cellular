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