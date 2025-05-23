"""Command-line interface for Conway's Game of Life."""

import argparse
import sys
import time
from typing import Optional, Tuple

from ..core.grid import Grid
from ..core.game import GameOfLife
from ..core.patterns import PatternLibrary


class CLIGameOfLife:
    """Command-line interface for running Game of Life simulations."""

    def __init__(self):
        """Initialize CLI interface."""
        self.pattern_library = PatternLibrary()

    def run_simulation(
        self,
        width: int,
        height: int,
        population_rate: float,
        toroidal: bool,
        max_generations: int,
        pattern: Optional[str] = None,
        pattern_x: int = 0,
        pattern_y: int = 0,
        verbose: bool = False,
        show_grid: bool = False,
    ) -> Tuple[int, str, dict]:
        """Run a Game of Life simulation.

        Args:
            width: Grid width
            height: Grid height
            population_rate: Initial random population rate (0.0-1.0)
            toroidal: Whether grid edges wrap around
            max_generations: Maximum generations to run
            pattern: Optional pattern name to load
            pattern_x: X offset for pattern placement
            pattern_y: Y offset for pattern placement
            verbose: Print progress updates
            show_grid: Show initial and final grid states

        Returns:
            Tuple of (final_generation, finish_reason, statistics)
        """
        # Create grid and game
        grid = Grid(width, height, wrap_edges=toroidal)
        game = GameOfLife(grid)

        if verbose:
            print(f"Initializing {width}x{height} grid (toroidal: {toroidal})")

        # Set up initial state
        if pattern:
            # Load specified pattern
            loaded_pattern = self.pattern_library.get_pattern(pattern)
            if loaded_pattern:
                if verbose:
                    print(f"Loading pattern '{pattern}' at ({pattern_x}, {pattern_y})")
                loaded_pattern.apply_to_grid(grid, pattern_x, pattern_y)
            else:
                print(
                    f"Warning: Pattern '{pattern}' not found, using random population"
                )
                grid.randomize(population_rate)
        else:
            # Use random population
            if verbose:
                print(f"Generating random population (rate: {population_rate:.2%})")
            grid.randomize(population_rate)

        initial_population = game.population
        if verbose:
            print(f"Initial population: {initial_population} cells")

        if show_grid:
            print("\nInitial grid:")
            print(self._format_grid(grid))

        # Run simulation
        start_time = time.time()

        if verbose:
            print(f"\nRunning simulation (max {max_generations} generations)...")

        final_generation, reason = game.run_until_stable(max_generations)

        end_time = time.time()
        duration = end_time - start_time

        # Get final statistics
        stats = game.get_statistics()
        stats["duration_seconds"] = duration
        stats["generations_per_second"] = (
            final_generation / duration if duration > 0 else 0
        )
        stats["initial_population"] = initial_population

        if show_grid and reason != "extinction":
            print(f"\nFinal grid (generation {final_generation}):")
            print(self._format_grid(grid))

        return final_generation, reason, stats

    def _format_grid(self, grid: Grid, max_size: int = 50) -> str:
        """Format grid for display, truncating if too large.

        Args:
            grid: Grid to format
            max_size: Maximum dimension to display

        Returns:
            Formatted grid string
        """
        if grid.width > max_size or grid.height > max_size:
            return f"Grid too large to display ({grid.width}x{grid.height})"

        return str(grid)

    def list_patterns(self) -> None:
        """List available patterns by category."""
        categories = self.pattern_library.get_patterns_by_category()

        print("Available patterns:")
        for category, patterns in categories.items():
            print(f"\n{category}:")
            for pattern_name in patterns:
                pattern = self.pattern_library.get_pattern(pattern_name)
                if pattern:
                    size = pattern.get_size()
                    population = len(pattern.cells)
                    print(f"  {pattern_name}: {size[0]}x{size[1]}, {population} cells")
                    if pattern.description:
                        print(f"    {pattern.description}")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Run Conway's Game of Life simulations from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run random 50x50 simulation with 10% population
  cellular-cli --width 50 --height 50 --population 0.1

  # Run glider on 20x20 toroidal grid
  cellular-cli -W 20 -H 20 --pattern Glider --toroidal

  # Run R-pentomino with verbose output
  cellular-cli --pattern "R-pentomino" --verbose --show-grid

  # List available patterns
  cellular-cli --list-patterns
        """,
    )

    # Grid configuration
    parser.add_argument(
        "-W", "--width", type=int, default=50, help="Grid width (default: 50)"
    )

    parser.add_argument(
        "-H", "--height", type=int, default=50, help="Grid height (default: 50)"
    )

    parser.add_argument(
        "-p",
        "--population",
        type=float,
        default=0.1,
        help="Initial random population rate 0.0-1.0 (default: 0.1)",
    )

    parser.add_argument(
        "-t",
        "--toroidal",
        action="store_true",
        help="Enable toroidal (wraparound) edges",
    )

    # Pattern configuration
    parser.add_argument(
        "--pattern",
        type=str,
        help="Load a specific pattern instead of random population",
    )

    parser.add_argument(
        "--pattern-x",
        type=int,
        default=0,
        help="X offset for pattern placement (default: 0)",
    )

    parser.add_argument(
        "--pattern-y",
        type=int,
        default=0,
        help="Y offset for pattern placement (default: 0)",
    )

    # Simulation configuration
    parser.add_argument(
        "-m",
        "--max-generations",
        type=int,
        default=10000,
        help="Maximum generations to simulate (default: 10000)",
    )

    # Output configuration
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "-g",
        "--show-grid",
        action="store_true",
        help="Display initial and final grid states (small grids only)",
    )

    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List all available patterns and exit",
    )

    return parser


def format_finish_reason(reason: str, stats: dict) -> str:
    """Format the simulation finish reason for display.

    Args:
        reason: Finish reason from GameOfLife.run_until_stable
        stats: Statistics dictionary

    Returns:
        Formatted reason string
    """
    if reason == "extinction":
        return "Extinction - all cells died"
    elif reason == "cycle":
        cycle_len = stats.get("cycle_length", 0)
        cycle_start = stats.get("cycle_start_generation", 0)
        return (
            f"Cycle detected - length {cycle_len}, started at generation {cycle_start}"
        )
    elif reason == "max_generations":
        return f"Maximum generations reached ({stats.get('generation', 0)})"
    else:
        return f"Unknown reason: {reason}"


def print_results(
    final_generation: int, reason: str, stats: dict, verbose: bool
) -> None:
    """Print simulation results.

    Args:
        final_generation: Final generation number
        reason: Finish reason
        stats: Statistics dictionary
        verbose: Whether to show detailed statistics
    """
    print(f"\nSimulation completed after {final_generation} generations")
    print(f"Finish reason: {format_finish_reason(reason, stats)}")

    if verbose:
        print("\nDetailed Statistics:")
        print(f"  Grid size: {stats['grid_size'][0]}x{stats['grid_size'][1]}")
        print(f"  Initial population: {stats['initial_population']}")
        print(f"  Final population: {stats['population']}")
        print(f"  Population density: {stats['population_density']:.2%}")
        print(f"  Population change rate: {stats['population_change_rate']:.2f}")
        print(f"  Duration: {stats['duration_seconds']:.3f} seconds")
        print(f"  Speed: {stats['generations_per_second']:.0f} generations/second")

        if stats["bounding_box"]:
            bbox = stats["bounding_box"]
            bbox_size = stats["bounding_box_size"]
            print(
                f"  Bounding box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]}) "
                f"[{bbox_size[0]}x{bbox_size[1]}]"
            )
    else:
        # Compact summary
        initial_pop = stats["initial_population"]
        final_pop = stats["population"]
        duration = stats["duration_seconds"]
        speed = stats["generations_per_second"]

        print(
            "Population: {} → {}, "
            "Duration: {:.3f}s, "
            "Speed: {:.0f} gen/s".format(initial_pop, final_pop, duration, speed)
        )


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command-line arguments.

    Args:
        args: Parsed arguments

    Returns:
        True if arguments are valid
    """
    errors = []

    if args.width <= 0:
        errors.append("Width must be positive")

    if args.height <= 0:
        errors.append("Height must be positive")

    if not 0.0 <= args.population <= 1.0:
        errors.append("Population rate must be between 0.0 and 1.0")

    if args.max_generations <= 0:
        errors.append("Max generations must be positive")

    if args.pattern_x < 0:
        errors.append("Pattern X offset must be non-negative")

    if args.pattern_y < 0:
        errors.append("Pattern Y offset must be non-negative")

    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def main() -> int:
    """Main entry point for CLI interface.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    cli = CLIGameOfLife()

    # Handle special commands
    if args.list_patterns:
        cli.list_patterns()
        return 0

    # Validate arguments
    if not validate_args(args):
        return 1

    # Check if pattern exists
    if args.pattern:
        pattern = cli.pattern_library.get_pattern(args.pattern)
        if not pattern:
            available = cli.pattern_library.list_patterns()
            print(f"Error: Pattern '{args.pattern}' not found")
            print(f"Available patterns: {', '.join(available)}")
            print("Use --list-patterns to see detailed information")
            return 1

        # Auto-center pattern if no offset specified
        if args.pattern_x == 0 and args.pattern_y == 0:
            pattern_size = pattern.get_size()
            args.pattern_x = max(0, (args.width - pattern_size[0]) // 2)
            args.pattern_y = max(0, (args.height - pattern_size[1]) // 2)
            if args.verbose:
                print(f"Auto-centering pattern at ({args.pattern_x}, {args.pattern_y})")

    try:
        # Run simulation
        final_generation, reason, stats = cli.run_simulation(
            width=args.width,
            height=args.height,
            population_rate=args.population,
            toroidal=args.toroidal,
            max_generations=args.max_generations,
            pattern=args.pattern,
            pattern_x=args.pattern_x,
            pattern_y=args.pattern_y,
            verbose=args.verbose,
            show_grid=args.show_grid,
        )

        # Display results
        print_results(final_generation, reason, stats, args.verbose)

        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
