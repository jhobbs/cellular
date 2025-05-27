"""Command-line interface for Conway's Game of Life."""

import argparse
import sys
import time
import random
import json
import os
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable
from datetime import datetime

from ..core.grid import Grid
from ..core.game import GameOfLife
from ..core.batch_grid import BatchGrid
from ..core.batch_game import BatchGameOfLife
from ..core.patterns import PatternLibrary, Pattern


# Removed multiprocessing worker function - now using tensor-based parallelization


class CLIGameOfLife:
    """Command-line interface for running Game of Life simulations."""

    def __init__(self):
        """Initialize CLI interface."""
        self.pattern_library = PatternLibrary()

        # Load found patterns from CLI searches
        self._load_found_patterns()

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
        return_initial_grid: bool = False,
    ) -> Tuple[int, str, dict, Optional[Grid]]:
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
                print(f"Warning: Pattern '{pattern}' not found, using random population")
                grid.randomize(population_rate)
        else:
            # Use random population
            if verbose:
                print(f"Generating random population (rate: {population_rate:.2%})")
            grid.randomize(population_rate)

        initial_population = game.population

        # Save initial grid state if requested
        initial_grid_copy = None
        if return_initial_grid:
            initial_grid_copy = Grid(width, height, wrap_edges=toroidal)
            for x in range(width):
                for y in range(height):
                    if grid.get_cell(x, y):
                        initial_grid_copy.set_cell(x, y, True)

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
        stats["generations_per_second"] = final_generation / duration if duration > 0 else 0
        stats["initial_population"] = initial_population

        if show_grid and reason != "extinction":
            print(f"\nFinal grid (generation {final_generation}):")
            print(self._format_grid(grid))

        if return_initial_grid:
            return final_generation, reason, stats, initial_grid_copy
        else:
            return final_generation, reason, stats

    def parallel_search_for_condition(
        self,
        width: int,
        height: int,
        population_rate: float,
        toroidal: bool,
        max_generations: int,
        condition_type: str,
        condition_value: Any,
        condition_name: str,
        max_attempts: int = 1000,
        pattern: Optional[str] = None,
        pattern_x: int = 0,
        pattern_y: int = 0,
        verbose: bool = False,
        seed: Optional[int] = None,
        save_pattern: Optional[str] = None,
        device: str = 'cpu',
        batch_size: int = 0,
        continue_search: bool = False,
    ):
        """Search for patterns using tensor-based parallel processing.
        
        Args:
            width: Grid width
            height: Grid height
            population_rate: Initial random population rate
            toroidal: Whether edges wrap around
            max_generations: Max generations per simulation
            condition_type: Type of condition to search for
            condition_value: Value for the condition
            condition_name: Human-readable condition name
            max_attempts: Maximum number of attempts
            pattern: Optional pattern to use as base
            pattern_x: X offset for pattern
            pattern_y: Y offset for pattern
            verbose: Print progress updates
            seed: Random seed for reproducibility
            save_pattern: Base filename to save found patterns
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Number of parallel simulations (0 for auto)
            continue_search: Continue after finding first match
        """
        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            if verbose:
                print("CUDA not available, falling back to CPU")
            device = 'cpu'
            
        # Auto-determine batch size based on device and memory
        if batch_size <= 0:
            if device == 'cuda':
                # Conservative estimate for GPU memory
                batch_size = min(1000, max_attempts)
            else:
                # For CPU, use a reasonable batch size
                batch_size = min(100, max_attempts)
                
        # Adjust batch size to not exceed max_attempts
        batch_size = min(batch_size, max_attempts)
        
        if verbose:
            print(f"Tensor-based parallel search on {device.upper()}")
            print(f"Searching for condition: {condition_name}")
            print(f"Grid: {width}x{height} (toroidal: {toroidal})")
            print(f"Batch size: {batch_size} parallel simulations")
            print(f"Max attempts: {max_attempts}")
            print(f"Max generations per attempt: {max_generations}")
            
        # Initialize statistics
        start_time = time.time()
        total_attempts = 0
        found_patterns = []
        
        end_state_stats = {
            "cycles": {},
            "extinctions": 0,
            "max_generations": 0,
            "total_attempts": 0,
            "total_generations": 0,
        }
        
        # Process in batches
        remaining_attempts = max_attempts
        attempt_offset = 0
        
        while remaining_attempts > 0 and (continue_search or len(found_patterns) == 0):
            current_batch_size = min(batch_size, remaining_attempts)
            
            # Create batch grid and game
            batch_grid = BatchGrid(current_batch_size, width, height, toroidal, device)
            batch_game = BatchGameOfLife(batch_grid)
            
            # Initialize grids
            if pattern:
                # Load pattern for each grid
                loaded_pattern = self.pattern_library.get_pattern(pattern)
                if loaded_pattern:
                    # Apply pattern to each grid in batch
                    for i in range(current_batch_size):
                        temp_grid = Grid(width, height, toroidal)
                        loaded_pattern.apply_to_grid(temp_grid, pattern_x, pattern_y)
                        # Convert to tensor and copy to batch
                        cells_tensor = torch.from_numpy(temp_grid.cells.T).to(torch.uint8)
                        batch_grid.copy_from_single(i, cells_tensor)
                else:
                    # Pattern not found, use random
                    seeds = None
                    if seed is not None:
                        seeds = torch.arange(current_batch_size) + seed + attempt_offset
                    batch_grid.randomize(population_rate, seeds)
            else:
                # Random initialization
                seeds = None
                if seed is not None:
                    seeds = torch.arange(current_batch_size) + seed + attempt_offset
                batch_grid.randomize(population_rate, seeds)
                
            # Save initial states for pattern saving
            initial_states = []
            initial_populations = []
            for i in range(current_batch_size):
                initial_grid = Grid(width, height, toroidal)
                cells = batch_grid.extract_single(i).cpu().numpy()
                # Transpose back from (height, width) to (width, height)
                initial_grid._cells = cells.T.astype(np.int8)
                initial_states.append(initial_grid)
                initial_populations.append(int(batch_grid.populations[i]))
                
            # Run simulations
            if verbose and total_attempts % 100 == 0 and total_attempts > 0:
                elapsed = time.time() - start_time
                rate = total_attempts / elapsed if elapsed > 0 else 0
                print(f"Progress: {total_attempts}/{max_attempts} attempts - {rate:.1f} attempts/sec")
                
            batch_game.run_until_stable_batch(max_generations)
            
            # Get results and check conditions
            stats_list = batch_game.get_statistics_batch()
            reasons = batch_game.get_termination_reasons()
            
            # Update end state statistics
            for i in range(current_batch_size):
                end_state_stats["total_attempts"] += 1
                end_state_stats["total_generations"] += stats_list[i]["generation"]
                
                if reasons[i] == "cycle":
                    cycle_len = stats_list[i]["cycle_length"]
                    end_state_stats["cycles"][cycle_len] = end_state_stats["cycles"].get(cycle_len, 0) + 1
                elif reasons[i] == "extinction":
                    end_state_stats["extinctions"] += 1
                elif reasons[i] == "max_generations":
                    end_state_stats["max_generations"] += 1
                    
            # Check each simulation for condition match
            for i in range(current_batch_size):
                attempt_num = attempt_offset + i
                stats = stats_list[i]
                reason = reasons[i]
                
                # Add initial population and missing stats
                stats["initial_population"] = initial_populations[i]
                stats["population_change_rate"] = 0.0  # Not tracked in batch mode
                stats["duration_seconds"] = 0.0  # Will be set for successful matches
                stats["generations_per_second"] = 0.0  # Will be set for successful matches
                
                # Check condition
                condition_met = self._check_condition(
                    condition_type, condition_value, reason, stats, stats.get("generation", 0)
                )
                
                if condition_met:
                    # Found a match!
                    found_pattern_data = {
                        "attempt": attempt_num,
                        "final_gen": stats["generation"],
                        "reason": reason,
                        "stats": stats,
                        "initial_grid": initial_states[i],
                    }
                    found_patterns.append(found_pattern_data)
                    
                    # Save pattern if requested
                    if save_pattern:
                        pattern_num = len(found_patterns)
                        saved_file = self._save_successful_pattern(
                            initial_states[i],
                            f"{save_pattern}_{pattern_num}" if continue_search else save_pattern,
                            condition_name,
                            stats["generation"],
                            reason,
                            stats,
                            attempt_num,
                            width,
                            height,
                            population_rate,
                            toroidal,
                            max_generations,
                            pattern,
                            pattern_x,
                            pattern_y,
                            seed,
                        )
                        if saved_file:
                            details = []
                            if reason == "cycle" and stats.get("cycle_length"):
                                details.append(f"cycle length {stats['cycle_length']}")
                            details.append(f"attempt {attempt_num}")
                            print(f"💾 Saved pattern #{pattern_num}: {saved_file} ({', '.join(details)})")
                            
                    if not continue_search:
                        # Stop searching after first match
                        elapsed = time.time() - start_time
                        if verbose:
                            print(f"✓ Found matching configuration! Attempt {attempt_num}")
                            print(f"Search completed in {elapsed:.2f}s")
                            
                        stats["search_duration_seconds"] = elapsed
                        total_attempts = attempt_offset + i + 1
                        
                        return True, total_attempts, (stats["generation"], reason, stats, saved_file if save_pattern else None), end_state_stats
                        
            # Update counters
            total_attempts += current_batch_size
            remaining_attempts -= current_batch_size
            attempt_offset += current_batch_size
            
        # Search complete
        elapsed = time.time() - start_time
        
        if found_patterns:
            # Found patterns in continue mode
            if verbose:
                print(f"\n🎯 Found {len(found_patterns)} matching patterns")
                print(f"Total attempts: {total_attempts}")
                print(f"Search completed in {elapsed:.2f}s")
                rate = total_attempts / elapsed if elapsed > 0 else 0
                print(f"Search rate: {rate:.1f} attempts/second")
                
            # Return first pattern as primary result
            first = found_patterns[0]
            first["stats"]["search_duration_seconds"] = elapsed
            return True, total_attempts, (first["final_gen"], first["reason"], first["stats"], None), end_state_stats
        else:
            # No matches found
            if verbose:
                print(f"✗ No matching configuration found after {total_attempts} attempts")
                print(f"Search completed in {elapsed:.2f}s")
                rate = total_attempts / elapsed if elapsed > 0 else 0
                print(f"Search rate: {rate:.1f} attempts/second")
                
            return False, total_attempts, {"duration_seconds": elapsed}, end_state_stats
            
    def _check_condition(self, condition_type: str, condition_value: Any, reason: str, stats: Dict, final_gen: int) -> bool:
        """Check if a simulation result meets the search condition."""
        if condition_type == "cycle_length":
            return reason == "cycle" and stats.get("cycle_length") == condition_value
        elif condition_type == "runs_for_at_least":
            return final_gen >= condition_value and reason == "cycle"
        elif condition_type == "extinction_after":
            return reason == "extinction" and final_gen >= condition_value
        elif condition_type == "any_cycle_excluding":
            return reason == "cycle" and stats.get("cycle_length", 0) not in condition_value
        elif condition_type == "composite":
            exclude_list, allow_extinction, allow_max_gens = condition_value
            if reason == "cycle":
                cycle_length = stats.get("cycle_length", 0)
                return cycle_length not in exclude_list
            elif reason == "extinction" and allow_extinction:
                return True
            elif reason == "max_generations" and allow_max_gens:
                return True
        return False

    def _save_successful_pattern(
        self,
        grid: Grid,
        base_filename: str,
        condition_name: str,
        final_generation: int,
        reason: str,
        stats: Dict[str, Any],
        attempt: int,
        width: int,
        height: int,
        population_rate: float,
        toroidal: bool,
        max_generations: int,
        pattern: Optional[str] = None,
        pattern_x: int = 0,
        pattern_y: int = 0,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """Save a successful pattern to file.

        Args:
            grid: Final grid state
            base_filename: Base filename for saving
            condition_name: Description of the condition that was met
            final_generation: Generation when condition was met
            reason: Reason simulation ended
            stats: Simulation statistics
            attempt: Attempt number when pattern was found
            width: Grid width
            height: Grid height
            population_rate: Initial random population rate
            toroidal: Whether grid edges wrap around
            max_generations: Maximum generations limit
            pattern: Optional pattern name that was used
            pattern_x: X offset for pattern placement
            pattern_y: Y offset for pattern placement
            seed: Random seed used for reproducibility

        Returns:
            Saved filename or None if save failed
        """
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create descriptive suffix based on finish reason
            if reason == "cycle" and stats.get("cycle_length"):
                reason_suffix = f"_cycle{stats['cycle_length']}"
            elif reason == "extinction":
                reason_suffix = "_extinct"
            elif reason == "max_generations":
                reason_suffix = "_maxgen"
            else:
                reason_suffix = ""

            # Create filename
            if base_filename.endswith(".json"):
                base_filename = base_filename[:-5]  # Remove .json extension

            filename = f"{base_filename}{reason_suffix}_{timestamp}.json"

            # Create save directory if it doesn't exist
            save_dir = "found_patterns"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filepath = os.path.join(save_dir, filename)

            # Create pattern from current grid state (normalized to start at 0,0)
            found_pattern = Pattern.from_grid(
                grid, f"Found {condition_name}", f"Pattern found after {attempt} attempts"
            )
            # Normalize the pattern to ensure it starts at (0,0)
            found_pattern = found_pattern.normalize()

            # Create comprehensive save data
            save_data = {
                "pattern": found_pattern.to_dict(),
                "search_info": {
                    "condition": condition_name,
                    "attempt_number": attempt,
                    "search_timestamp": timestamp,
                    "final_generation": final_generation,
                    "finish_reason": reason,
                },
                "simulation_parameters": {
                    "width": width,
                    "height": height,
                    "toroidal": toroidal,
                    "population_rate": population_rate,
                    "max_generations": max_generations,
                    "pattern": pattern,
                    "pattern_x": pattern_x,
                    "pattern_y": pattern_y,
                    "seed": seed,
                },
                "statistics": stats,
                "reload_command": self._generate_reload_command(
                    pattern, width, height, toroidal, population_rate, max_generations, pattern_x, pattern_y, seed
                ),
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2)

            # Also add to pattern library for immediate use
            self.pattern_library.add_pattern(found_pattern)

            return filepath

        except Exception as e:
            print(f"Warning: Failed to save pattern: {e}")
            return None

    def _generate_reload_command(
        self,
        pattern_name: Optional[str],
        width: int,
        height: int,
        toroidal: bool,
        population_rate: float,
        max_generations: int,
        pattern_x: int = 0,
        pattern_y: int = 0,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a command line to reload this exact simulation configuration."""
        cmd_parts = ["cellular-cli"]

        # Grid parameters
        cmd_parts.append(f"--width {width}")
        cmd_parts.append(f"--height {height}")
        if toroidal:
            cmd_parts.append("--toroidal")

        # Pattern or population
        if pattern_name:
            cmd_parts.append(f"--pattern '{pattern_name}'")
            if pattern_x != 0:
                cmd_parts.append(f"--pattern-x {pattern_x}")
            if pattern_y != 0:
                cmd_parts.append(f"--pattern-y {pattern_y}")
        else:
            cmd_parts.append(f"--population {population_rate}")

        # Simulation parameters
        if max_generations != 10000:  # Default value
            cmd_parts.append(f"--max-generations {max_generations}")

        # Reproducibility
        if seed is not None:
            cmd_parts.append(f"--seed {seed}")

        return " ".join(cmd_parts)

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
        pattern_index = 1
        for category, patterns in categories.items():
            print(f"\n{category}:")
            for pattern_name in patterns:
                pattern = self.pattern_library.get_pattern(pattern_name)
                if pattern:
                    size = pattern.get_size()
                    population = len(pattern.cells)
                    if category == "Found Patterns":
                        # Show index for found patterns for easier selection
                        print(f"  [{pattern_index}] {pattern_name}: {size[0]}x{size[1]}, {population} cells")
                        pattern_index += 1
                    else:
                        print(f"  {pattern_name}: {size[0]}x{size[1]}, {population} cells")
                    if pattern.description:
                        print(f"    {pattern.description}")

        if pattern_index > 1:
            print("\n💡 Tip: Use --pattern '[1]' to load found pattern by index number")

    def _load_found_patterns(self) -> None:
        """Load patterns from the found_patterns directory."""
        found_patterns_dir = "found_patterns"
        if not os.path.exists(found_patterns_dir):
            return

        try:
            for filename in os.listdir(found_patterns_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(found_patterns_dir, filename)
                    try:
                        with open(filepath, "r") as f:
                            saved_data = json.load(f)

                        # Extract pattern from saved data
                        if "pattern" in saved_data:
                            pattern_data = saved_data["pattern"]

                            # Create concise pattern name from condition and parameters
                            condition = saved_data.get("search_info", {}).get("condition", "found pattern")
                            sim_params = saved_data.get("simulation_parameters", {})

                            # Create a more readable, concise name
                            if "cycle length" in condition:
                                cycle_len = condition.split()[-1]
                                pattern_name = f"Cycle-{cycle_len}"
                            elif "runs for at least" in condition:
                                gens = condition.split()[-3]  # "runs for at least N generations"
                                pattern_name = f"Long-run {gens}+"
                            elif "extinction after" in condition:
                                gens = condition.split()[-3]  # "goes extinct after at least N generations"
                                pattern_name = f"Dies@{gens}+"
                            elif "population" in condition:
                                if "threshold" in condition:
                                    pop = condition.split()[-1]
                                    pattern_name = f"Pop≥{pop}"
                                else:  # stabilizes with population
                                    pop = condition.split()[-2]  # "stabilizes with exactly N cells"
                                    pattern_name = f"Stable@{pop}"
                            elif "bounding box" in condition:
                                size = condition.split()[-1]  # "has bounding box of at least WxH"
                                pattern_name = f"Box≥{size}"
                            else:
                                # Fallback for unknown conditions
                                pattern_name = (
                                    condition.replace("finishes with ", "")
                                    .replace("goes extinct after at least ", "Dies@")
                                    .replace(" generations", "")
                                )

                            # Add grid size and timestamp for uniqueness
                            width = sim_params.get("width", "?")
                            height = sim_params.get("height", "?")
                            timestamp = saved_data.get("search_info", {}).get("search_timestamp", "unknown")
                            # Use last 4 characters of timestamp for brevity
                            unique_id = timestamp[-4:] if len(timestamp) >= 4 else timestamp
                            pattern_name += f" ({width}×{height}) #{unique_id}"

                            # Create pattern object
                            pattern = Pattern(
                                name=pattern_name,
                                cells=pattern_data.get("cells", []),
                                description=f"Pattern found by CLI search: {condition}",
                                metadata={"category": "Found Patterns", "source_file": filename},
                            )

                            # Store simulation parameters for recreation
                            pattern.simulation_params = saved_data["simulation_parameters"]

                            # Add to pattern library
                            self.pattern_library.add_pattern(pattern)

                    except Exception as e:
                        print(f"Warning: Failed to load found pattern {filename}: {e}")

        except Exception as e:
            print(f"Warning: Failed to scan found_patterns directory: {e}")

        # Override pattern library categorization to include found patterns
        self._setup_found_patterns_category()

    def _setup_found_patterns_category(self) -> None:
        """Override pattern library categorization to include found patterns."""
        # Store original method
        original_get_patterns_by_category = self.pattern_library.get_patterns_by_category

        def enhanced_get_patterns_by_category():
            # Get original categories
            categories = original_get_patterns_by_category()

            # Move found patterns from Custom to Found Patterns category
            found_patterns = []
            custom_patterns = categories.get("Custom", [])

            # Find patterns that should be in Found Patterns category
            for pattern_name in self.pattern_library._patterns:
                pattern = self.pattern_library._patterns[pattern_name]
                if hasattr(pattern, "metadata") and pattern.metadata.get("category") == "Found Patterns":
                    found_patterns.append(pattern_name)
                    # Remove from custom category if it's there
                    if pattern_name in custom_patterns:
                        custom_patterns.remove(pattern_name)

            if found_patterns:
                categories["Found Patterns"] = found_patterns
                # Update custom category without found patterns
                if custom_patterns:
                    categories["Custom"] = custom_patterns
                elif "Custom" in categories:
                    del categories["Custom"]

            return categories

        # Replace the method
        self.pattern_library.get_patterns_by_category = enhanced_get_patterns_by_category


def create_condition_functions() -> Dict[str, Callable]:
    """Create predefined condition functions for search mode.

    Returns:
        Dictionary mapping condition names to functions
    """

    def runs_for_at_least_n_generations(min_generations: int):
        """Condition: simulation runs for at least N generations without cycling."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            if reason == "cycle":
                # Check if it ran long enough before cycling
                cycle_start = stats.get("cycle_start_generation", 0)
                return cycle_start >= min_generations
            elif reason == "max_generations":
                # Hit max generations without cycling - good
                return final_gen >= min_generations
            else:
                # Extinction - doesn't meet condition
                return False

        return condition

    def finishes_with_cycle_length(target_cycle_length: int):
        """Condition: simulation finishes with a specific cycle length."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            if reason == "cycle":
                return stats.get("cycle_length", 0) == target_cycle_length
            return False

        return condition

    def finishes_with_extinction_after_n_generations(min_generations: int):
        """Condition: goes extinct after at least N generations."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            return reason == "extinction" and final_gen >= min_generations

        return condition


    def finishes_with_any_cycle_excluding(exclude_cycles: list):
        """Condition: simulation finishes with any cycle length except those in exclude list."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            if reason == "cycle":
                cycle_length = stats.get("cycle_length", 0)
                return cycle_length not in exclude_cycles
            return False

        return condition

    def composite_condition(exclude_cycles: list, allow_extinction: bool, allow_max_gens: bool):
        """Composite condition: cycles (excluding list), extinction, or max generations."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            if reason == "cycle":
                cycle_length = stats.get("cycle_length", 0)
                return cycle_length not in exclude_cycles
            elif reason == "extinction" and allow_extinction:
                return True
            elif reason == "max_generations" and allow_max_gens:
                return True
            return False

        return condition

    return {
        "runs_for_at_least_n_generations": runs_for_at_least_n_generations,
        "finishes_with_cycle_length": finishes_with_cycle_length,
        "finishes_with_extinction_after_n_generations": (finishes_with_extinction_after_n_generations),
        "finishes_with_any_cycle_excluding": finishes_with_any_cycle_excluding,
        "composite_condition": composite_condition,
    }


def parse_search_condition(condition_str: str) -> Tuple[Callable, str]:
    """Parse a condition string into a condition function and description.

    Args:
        condition_str: String like "runs_for_at_least:100" or "cycle_length:3"

    Returns:
        Tuple of (condition_function, description)

    Raises:
        ValueError: If condition string is invalid
    """
    condition_funcs = create_condition_functions()

    if ":" not in condition_str:
        raise ValueError(f"Invalid condition format: '{condition_str}'. Expected 'type:value'")

    condition_type, value_str = condition_str.split(":", 1)

    try:
        if condition_type == "runs_for_at_least":
            value = int(value_str)
            func = condition_funcs["runs_for_at_least_n_generations"](value)
            desc = f"runs for at least {value} generations without cycling"

        elif condition_type == "cycle_length":
            value = int(value_str)
            func = condition_funcs["finishes_with_cycle_length"](value)
            desc = f"finishes with cycle length {value}"

        elif condition_type == "extinction_after":
            value = int(value_str)
            func = condition_funcs["finishes_with_extinction_after_n_generations"](value)
            desc = f"goes extinct after at least {value} generations"


        elif condition_type == "any_cycle_excluding":
            # Parse comma-separated list of cycle lengths to exclude
            exclude_list = [int(x.strip()) for x in value_str.split(",")]
            func = condition_funcs["finishes_with_any_cycle_excluding"](exclude_list)
            desc = f"finishes with any cycle except lengths: {', '.join(map(str, exclude_list))}"
            value = exclude_list

        elif condition_type == "composite":
            # Parse format: "composite:exclude_cycles;allow_extinction;allow_max_gens"
            # Example: "composite:1,2;true;true" excludes cycles 1,2 and allows extinction/max_gens
            parts = value_str.split(";")
            if len(parts) != 3:
                raise ValueError("composite requires format 'exclude_cycles;allow_extinction;allow_max_gens' (e.g., '1,2;true;true')")
            
            # Parse excluded cycles
            if parts[0].strip():
                exclude_list = [int(x.strip()) for x in parts[0].split(",")]
            else:
                exclude_list = []
            
            # Parse boolean flags
            allow_extinction = parts[1].strip().lower() == "true"
            allow_max_gens = parts[2].strip().lower() == "true"
            
            func = condition_funcs["composite_condition"](exclude_list, allow_extinction, allow_max_gens)
            
            # Build description
            desc_parts = []
            if exclude_list:
                desc_parts.append(f"cycles except {', '.join(map(str, exclude_list))}")
            else:
                desc_parts.append("any cycle")
            if allow_extinction:
                desc_parts.append("extinction")
            if allow_max_gens:
                desc_parts.append("max generations")
            desc = f"finishes with: {' or '.join(desc_parts)}"
            value = (exclude_list, allow_extinction, allow_max_gens)

        else:
            available = [
                "runs_for_at_least",
                "cycle_length",
                "extinction_after",
                "any_cycle_excluding",
                "composite",
            ]
            raise ValueError(f"Unknown condition type '{condition_type}'. " f"Available: {', '.join(available)}")

    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid numeric value in condition: '{value_str}'")
        raise

    return func, desc


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

  # Search for patterns with specific cycle lengths
  cellular-cli --search cycle_length:3 --search-attempts 500

  # Search for long-running patterns
  cellular-cli --search runs_for_at_least:1000 --population 0.15

  # Search using 4 workers (parallel)
  cellular-cli --search cycle_length:3 --workers 4

  # Force sequential search
  cellular-cli --search cycle_length:3 --workers 1

  # Search for any cycle except still lifes (cycle length 1) and blinkers (cycle length 2)
  cellular-cli --search any_cycle_excluding:1,2 --search-attempts 1000

  # Continue searching after finding matches, save all found patterns
  cellular-cli --search cycle_length:3 --continue-search --save-pattern cycles3 --search-attempts 500
  
  # Composite search: find cycles (except 1,2), extinctions, or max generation patterns
  cellular-cli --search composite:1,2;true;true --save-pattern interesting --search-attempts 1000
        """,
    )

    # Grid configuration
    parser.add_argument("-W", "--width", type=int, default=50, help="Grid width (default: 50)")

    parser.add_argument("-H", "--height", type=int, default=50, help="Grid height (default: 50)")

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

    # Search mode configuration
    parser.add_argument(
        "--search",
        type=str,
        help="Search for configurations meeting a condition " "(e.g., 'cycle_length:3', 'any_cycle_excluding:1,2')",
    )

    parser.add_argument(
        "--search-attempts",
        type=int,
        default=1000,
        help="Maximum attempts in search mode (default: 1000)",
    )

    parser.add_argument(
        "--search-seed",
        type=int,
        help="Random seed for reproducible searches",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tensor computations on (default: cpu)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Number of parallel simulations per batch (0 = auto-detect)",
    )

    parser.add_argument(
        "--save-pattern",
        type=str,
        help="Base filename to save successful search patterns (saved to found_patterns/ directory)",
    )

    parser.add_argument(
        "--continue-search",
        action="store_true",
        help="Continue searching through all max attempts even after finding matches (saves all found patterns)",
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
        return f"Cycle detected - length {cycle_len}, started at generation {cycle_start}"
    elif reason == "max_generations":
        return f"Maximum generations reached ({stats.get('generation', 0)})"
    else:
        return f"Unknown reason: {reason}"


def print_end_state_statistics(end_state_stats: dict, verbose: bool = True) -> None:
    """Print end state statistics showing how attempts concluded.

    Args:
        end_state_stats: Dictionary with cycle, extinction, and max_generations counts
        verbose: Whether to show detailed breakdown
    """
    if not end_state_stats or end_state_stats.get("total_attempts", 0) == 0:
        return

    total = end_state_stats["total_attempts"]
    total_gens = end_state_stats.get("total_generations", 0)
    cycles = end_state_stats.get("cycles", {})
    extinctions = end_state_stats.get("extinctions", 0)
    max_gens = end_state_stats.get("max_generations", 0)

    avg_gens = total_gens / total if total > 0 else 0
    print(f"\nEnd State Distribution ({total} attempts, {total_gens} generations, avg {avg_gens:.1f} gen/attempt):")

    # Show cycle breakdown
    if cycles:
        print("  Cycles:")
        # Sort cycle lengths for consistent display
        for cycle_length in sorted(cycles.keys()):
            count = cycles[cycle_length]
            percentage = (count / total) * 100
            print(f"    Length {cycle_length}: {count} attempts ({percentage:.1f}%)")

        total_cycles = sum(cycles.values())
        cycle_percentage = (total_cycles / total) * 100
        print(f"    Total cycles: {total_cycles} attempts ({cycle_percentage:.1f}%)")

    # Show other end states
    if extinctions > 0:
        percentage = (extinctions / total) * 100
        print(f"  Extinctions: {extinctions} attempts ({percentage:.1f}%)")

    if max_gens > 0:
        percentage = (max_gens / total) * 100
        print(f"  Hit max generations: {max_gens} attempts ({percentage:.1f}%)")


def print_results(final_generation: int, reason: str, stats: dict, verbose: bool) -> None:
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
        if "duration_seconds" in stats:
            print(f"  Duration: {stats['duration_seconds']:.3f} seconds")
            print(f"  Speed: {stats['generations_per_second']:.0f} generations/second")

        if stats["bounding_box"]:
            bbox = stats["bounding_box"]
            bbox_size = stats["bounding_box_size"]
            print(
                f"  Bounding box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]}) " f"[{bbox_size[0]}x{bbox_size[1]}]"
            )
    else:
        # Compact summary
        initial_pop = stats["initial_population"]
        final_pop = stats["population"]
        duration = stats.get("duration_seconds", 0)
        speed = stats.get("generations_per_second", 0)

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

    if hasattr(args, "search_attempts") and args.search_attempts <= 0:
        errors.append("Search attempts must be positive")

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

    # Check if pattern exists (handle both name and index)
    if args.pattern:
        pattern = None

        # Check if pattern is specified by index (e.g., "[1]")
        if args.pattern.startswith("[") and args.pattern.endswith("]"):
            try:
                index = int(args.pattern[1:-1])
                # Get found patterns in order
                categories = cli.pattern_library.get_patterns_by_category()
                found_patterns = categories.get("Found Patterns", [])
                if 1 <= index <= len(found_patterns):
                    pattern_name = found_patterns[index - 1]
                    pattern = cli.pattern_library.get_pattern(pattern_name)
                    # Update args.pattern to the resolved name for later use
                    args.pattern = pattern_name
                    if args.verbose:
                        print(f"Loading found pattern #{index}: {pattern_name}")
                else:
                    print(f"Error: Found pattern index {index} out of range (1-{len(found_patterns)})")
                    print("Use --list-patterns to see available found patterns")
                    return 1
            except ValueError:
                print(f"Error: Invalid pattern index format '{args.pattern}'. Use [1], [2], etc.")
                return 1
        else:
            # Try to find pattern by name
            pattern = cli.pattern_library.get_pattern(args.pattern)

        if not pattern:
            available = cli.pattern_library.list_patterns()
            print(f"Error: Pattern '{args.pattern}' not found")
            print(f"Available patterns: {', '.join(available)}")
            print("Use --list-patterns to see detailed information")
            return 1

        # For found patterns, use their original simulation parameters unless overridden
        if hasattr(pattern, "simulation_params") and pattern.simulation_params:
            sim_params = pattern.simulation_params

            # Use original grid size if not explicitly overridden by user
            if not any("--width" in arg or "-W" in arg for arg in sys.argv):
                args.width = sim_params.get("width", args.width)
            if not any("--height" in arg or "-H" in arg for arg in sys.argv):
                args.height = sim_params.get("height", args.height)
            if not any("--toroidal" in arg or "-t" in arg for arg in sys.argv):
                args.toroidal = sim_params.get("toroidal", args.toroidal)

            if args.verbose:
                print("Using found pattern's original parameters:")
                print(f"  Grid: {args.width}×{args.height} (toroidal: {args.toroidal})")
                if sim_params.get("pattern"):
                    print(f"  Based on pattern: {sim_params['pattern']}")
                else:
                    print(f"  Population rate: {sim_params.get('population_rate', 'unknown')}")

        # Auto-center pattern if no offset specified
        if args.pattern_x == 0 and args.pattern_y == 0:
            pattern_size = pattern.get_size()
            args.pattern_x = max(0, (args.width - pattern_size[0]) // 2)
            args.pattern_y = max(0, (args.height - pattern_size[1]) // 2)
            if args.verbose:
                print(f"Auto-centering pattern at ({args.pattern_x}, {args.pattern_y})")

    try:
        # Handle search mode
        if args.search:
            try:
                condition_func, condition_desc = parse_search_condition(args.search)
                # Parse condition type and value for parallel search
                condition_type, value_str = args.search.split(":", 1)
                if condition_type == "any_cycle_excluding":
                    # Parse comma-separated list of cycle lengths to exclude
                    condition_value = [int(x.strip()) for x in value_str.split(",")]
                elif condition_type == "composite":
                    # Parse composite format
                    parts = value_str.split(";")
                    if len(parts) != 3:
                        raise ValueError("composite requires format 'exclude_cycles;allow_extinction;allow_max_gens'")
                    # Parse excluded cycles
                    if parts[0].strip():
                        exclude_list = [int(x.strip()) for x in parts[0].split(",")]
                    else:
                        exclude_list = []
                    # Parse boolean flags
                    allow_extinction = parts[1].strip().lower() == "true"
                    allow_max_gens = parts[2].strip().lower() == "true"
                    condition_value = (exclude_list, allow_extinction, allow_max_gens)
                else:
                    condition_value = int(value_str)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

            # Start overall timing
            overall_start_time = time.time()

            # Use tensor-based parallel search
            found, attempts, result, end_state_stats = cli.parallel_search_for_condition(
                width=args.width,
                height=args.height,
                population_rate=args.population,
                toroidal=args.toroidal,
                max_generations=args.max_generations,
                condition_type=condition_type,
                condition_value=condition_value,
                condition_name=condition_desc,
                max_attempts=args.search_attempts,
                pattern=args.pattern,
                pattern_x=args.pattern_x,
                pattern_y=args.pattern_y,
                verbose=args.verbose,
                seed=args.search_seed,
                save_pattern=args.save_pattern,
                device=args.device,
                batch_size=args.batch_size,
                continue_search=args.continue_search,
            )

            # Calculate overall runtime including overhead
            overall_elapsed = time.time() - overall_start_time

            if found and result:
                final_generation, reason, stats, saved_file = result
                search_duration = stats.get("search_duration_seconds", 0)

                # Use total attempts from end_state_stats for accurate rate calculation
                total_work_attempts = end_state_stats.get("total_attempts", attempts) if end_state_stats else attempts
                # For single success, estimate total generations: attempts * final_generation (rough estimate)
                if end_state_stats:
                    total_generations = end_state_stats.get("total_generations", 0)
                else:
                    # Rough estimate: assume all attempts ran similar number of generations
                    total_generations = attempts * final_generation
                attempts_per_sec = total_work_attempts / search_duration if search_duration > 0 else 0

                # Calculate average generations per attempt for computational rate
                avg_gens_per_attempt = total_generations / total_work_attempts if total_work_attempts > 0 else 0

                overhead = overall_elapsed - search_duration

                # Simple wall-clock generations per second
                wall_clock_gens_per_sec = total_generations / search_duration if search_duration > 0 else 0

                print(f"\n🎯 SUCCESS: Found configuration after {attempts} attempts")
                print(f"Total work: {total_work_attempts} attempts across all workers")
                print(
                    f"Search rate: {attempts_per_sec:.1f} attempts/second, "
                    f"{wall_clock_gens_per_sec:.1f} generations/second"
                )
                print(f"Total generations: {total_generations} (avg {avg_gens_per_attempt:.1f} gen/attempt)")
                print(
                    f"Search time: {search_duration:.2f}s, Overall runtime: {overall_elapsed:.2f}s "
                    f"(overhead: {overhead:.2f}s)"
                )
                print_results(final_generation, reason, stats, args.verbose)
                if saved_file:
                    # Include relevant details in save message
                    details = []
                    # Include cycle length for any cycle-based condition
                    if reason == "cycle" and stats.get("cycle_length"):
                        details.append(f"cycle length {stats['cycle_length']}")
                    details.append(f"attempt {attempts}")
                    print(f"💾 Pattern saved to: {saved_file} ({', '.join(details)})")

                # Show end state statistics
                print_end_state_statistics(end_state_stats, args.verbose)
                return 0
            else:
                # Get duration from failed search result
                search_duration = result.get("duration_seconds", 0) if result else 0

                # Use total attempts from end_state_stats for accurate rate calculation
                total_work_attempts = end_state_stats.get("total_attempts", attempts) if end_state_stats else attempts
                # For failed searches, we don't have meaningful generation data without end_state_stats
                if end_state_stats:
                    total_generations = end_state_stats.get("total_generations", 0)
                else:
                    # Without end state stats, we can't estimate total generations for failed searches
                    total_generations = 0
                attempts_per_sec = total_work_attempts / search_duration if search_duration > 0 else 0

                # Calculate average generations per attempt for computational rate
                avg_gens_per_attempt = total_generations / total_work_attempts if total_work_attempts > 0 else 0

                overhead = overall_elapsed - search_duration

                # Determine if this was an interruption based on actual vs expected attempts
                was_interrupted = attempts < args.search_attempts and search_duration > 0

                if was_interrupted:
                    print(f"\n⚠️ INTERRUPTED: Search stopped after {attempts} attempts (of {args.search_attempts})")
                    print(f"Total work: {total_work_attempts} attempts across all workers")
                else:
                    print(f"\n❌ FAILED: No configuration found after {attempts} attempts")
                    print(f"Total work: {total_work_attempts} attempts across all workers")

                if search_duration > 0:
                    # Simple wall-clock generations per second
                    wall_clock_gens_per_sec = total_generations / search_duration if search_duration > 0 else 0

                    print(
                        f"Search rate: {attempts_per_sec:.1f} attempts/second, "
                        f"{wall_clock_gens_per_sec:.1f} generations/second"
                    )
                    print(f"Total generations: {total_generations} (avg {avg_gens_per_attempt:.1f} gen/attempt)")
                    print(
                        f"Search time: {search_duration:.2f}s, Overall runtime: {overall_elapsed:.2f}s "
                        f"(overhead: {overhead:.2f}s)"
                    )
                else:
                    print(f"Overall runtime: {overall_elapsed:.2f}s")

                print(f"Condition: {condition_desc}")
                if not was_interrupted:
                    print("Try increasing --search-attempts or adjusting parameters")

                # Show end state statistics
                print_end_state_statistics(end_state_stats, args.verbose)
                return 1

        else:
            # Run single simulation
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
