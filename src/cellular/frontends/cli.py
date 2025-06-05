"""Command-line interface for Conway's Game of Life."""

import argparse
import sys
import time
import random
import json
import os
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from typing import Optional, Tuple, Dict, Any, Callable
from datetime import datetime

from ..core.grid import Grid
from ..core.game import GameOfLife
from ..core.patterns import PatternLibrary, Pattern
from ..core.bulk_runner import BulkSimulationRunner, SimulationConfig


def _search_worker_batched(args_tuple):
    """Worker function for batched parallel search - must be at module level for pickling."""
    (
        width,
        height,
        population_rate,
        toroidal,
        max_generations,
        pattern,
        pattern_x,
        pattern_y,
        condition_type,
        condition_value,
        seed_base,
        worker_id,
        work_queue,
        stop_flag,
        batch_size,
        progress_counter,
        progress_lock,
        continue_search,
        found_patterns_queue,
        topology,
    ) = args_tuple

    # Create local pattern library for this worker
    pattern_library = PatternLibrary()
    worker_attempts = 0

    # Track end state statistics for this worker
    end_state_stats = {
        "cycles": {},  # cycle_length -> count
        "extinctions": 0,
        "max_generations": 0,
        "total_attempts": 0,
        "total_generations": 0,  # Track total generations computed
    }

    try:
        while not stop_flag.value:
            # Get a batch of work
            try:
                batch_start = work_queue.get_nowait()
            except Exception:
                # No more work available
                break

            # Process this batch
            for batch_offset in range(batch_size):
                # Check if another worker found a match
                if stop_flag.value:
                    break

                attempt = batch_start + batch_offset
                worker_attempts += 1

                # Update global progress counter
                with progress_lock:
                    progress_counter.value += 1

                # Unique seed for this attempt
                attempt_seed = seed_base + attempt if seed_base else None
                if attempt_seed:
                    random.seed(attempt_seed)

                try:
                    # Create grid and game
                    grid = Grid(width, height, wrap_edges=toroidal, topology=topology)
                    game = GameOfLife(grid)

                    # Set up initial state
                    if pattern:
                        loaded_pattern = pattern_library.get_pattern(pattern)
                        if loaded_pattern:
                            loaded_pattern.apply_to_grid(grid, pattern_x, pattern_y)
                        else:
                            grid.randomize(population_rate)
                    else:
                        grid.randomize(population_rate)

                    # Save initial grid state for pattern saving
                    initial_grid = Grid(width, height, toroidal)
                    for x in range(width):
                        for y in range(height):
                            if grid.get_cell(x, y):
                                initial_grid.set_cell(x, y, True)

                    # Run simulation with timing
                    initial_population = game.population
                    sim_start_time = time.time()
                    final_gen, reason = game.run_until_stable(max_generations)
                    sim_end_time = time.time()
                    sim_duration = sim_end_time - sim_start_time

                    stats = game.get_statistics()
                    stats["initial_population"] = initial_population
                    stats["duration_seconds"] = sim_duration
                    stats["generations_per_second"] = final_gen / sim_duration if sim_duration > 0 else 0

                    # Track end state statistics
                    end_state_stats["total_attempts"] += 1
                    end_state_stats["total_generations"] += final_gen
                    if reason == "cycle":
                        cycle_length = stats.get("cycle_length", 0)
                        end_state_stats["cycles"][cycle_length] = end_state_stats["cycles"].get(cycle_length, 0) + 1
                    elif reason == "extinction":
                        end_state_stats["extinctions"] += 1
                    elif reason == "max_generations":
                        end_state_stats["max_generations"] += 1

                    # Check if condition is met based on type
                    condition_met = False
                    if condition_type == "cycle_length":
                        condition_met = reason == "cycle" and stats.get("cycle_length") == condition_value
                    elif condition_type == "runs_for_at_least":
                        condition_met = final_gen >= condition_value and reason == "cycle"
                    elif condition_type == "extinction_after":
                        condition_met = reason == "extinction" and final_gen >= condition_value
                    elif condition_type == "population_threshold":
                        population_history = stats.get("population_history", [])
                        condition_met = any(pop >= condition_value for pop in population_history)
                    elif condition_type == "stabilizes_with_population":
                        condition_met = stats.get("population", 0) == condition_value and reason in (
                            "cycle",
                            "extinction",
                        )
                    elif condition_type == "bounding_box_size":
                        bbox_size = stats.get("bounding_box_size", (0, 0))
                        width_req, height_req = condition_value
                        condition_met = bbox_size[0] >= width_req and bbox_size[1] >= height_req
                    elif condition_type == "any_cycle_excluding":
                        condition_met = reason == "cycle" and stats.get("cycle_length", 0) not in condition_value
                    elif condition_type == "composite":
                        exclude_list, allow_extinction, allow_max_gens = condition_value
                        if reason == "cycle":
                            cycle_length = stats.get("cycle_length", 0)
                            condition_met = cycle_length not in exclude_list
                        elif reason == "extinction" and allow_extinction:
                            condition_met = True
                        elif reason == "max_generations" and allow_max_gens:
                            condition_met = True

                    if condition_met:
                        # Store the found pattern
                        found_pattern_data = {
                            "worker_id": worker_id,
                            "attempt": attempt,
                            "final_gen": final_gen,
                            "reason": reason,
                            "stats": stats,
                            "initial_grid": initial_grid,
                        }
                        found_patterns_queue.put(found_pattern_data)
                        
                        if not continue_search:
                            stop_flag.value = True  # Signal other workers to stop
                            return (
                                True,
                                worker_id,
                                attempt,
                                final_gen,
                                reason,
                                stats,
                                initial_grid,
                                worker_attempts,
                                end_state_stats,
                            )
                        # If continue_search is True, just keep going

                except Exception:
                    continue  # Skip failed attempts

    except Exception:
        pass  # Worker cleanup

    return False, worker_id, None, None, None, None, None, worker_attempts, end_state_stats


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
        topology: Optional[str] = None,
    ) -> Tuple[int, str, dict, Optional[Grid]]:
        """Run a Game of Life simulation.

        Args:
            width: Grid width
            height: Grid height
            population_rate: Initial random population rate (0.0-1.0)
            toroidal: DEPRECATED - use topology instead
            max_generations: Maximum generations to run
            pattern: Optional pattern name to load
            pattern_x: X offset for pattern placement
            pattern_y: Y offset for pattern placement
            verbose: Print progress updates
            show_grid: Show initial and final grid states
            topology: Grid edge topology ("toroidal", "bounded", "mirroring")

        Returns:
            Tuple of (final_generation, finish_reason, statistics)
        """
        # Create grid and game
        grid = Grid(width, height, wrap_edges=toroidal, topology=topology)
        game = GameOfLife(grid)

        if verbose:
            actual_topology = topology if topology else ("toroidal" if toroidal else "bounded")
            print(f"Initializing {width}x{height} grid (topology: {actual_topology})")

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
            initial_grid_copy = Grid(width, height, wrap_edges=toroidal, topology=topology)
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
        workers: int = 0,
        batch_size: int = 0,
        continue_search: bool = False,
        topology: Optional[str] = None,
    ):
        """Search for patterns using batched parallel processing with work stealing."""
        if workers <= 0:
            workers = mp.cpu_count()

        # Auto-determine batch size based on expected work complexity
        if batch_size <= 0:
            # Aim for ~100-500 batches total to balance overhead vs load balancing
            target_batches = min(500, max(100, max_attempts // 10))
            batch_size = max(1, max_attempts // target_batches)

        total_batches = (max_attempts + batch_size - 1) // batch_size  # Ceiling division
        actual_max_attempts = total_batches * batch_size  # May be slightly more than requested

        if verbose:
            print(f"Batched parallel search using {workers} workers")
            print(f"Searching for condition: {condition_name}")
            actual_topology = topology if topology else ("toroidal" if toroidal else "bounded")
            print(f"Grid: {width}x{height} (topology: {actual_topology})")
            print(f"Max attempts: {max_attempts} (rounded up to {actual_max_attempts} for batching)")
            print(f"Max generations per attempt: {max_generations}")
            print(f"Batch size: {batch_size} attempts/batch ({total_batches} total batches)")

        # Create shared resources for work distribution
        with Manager() as manager:
            stop_flag = manager.Value("b", False)
            work_queue = manager.Queue()
            progress_counter = manager.Value("i", 0)  # Shared progress counter
            progress_lock = manager.Lock()  # Lock for progress counter
            found_patterns_queue = manager.Queue()  # Queue for found patterns in continue mode

            # Fill work queue with batch starting indices
            for batch_idx in range(total_batches):
                work_queue.put(batch_idx * batch_size)

            # Prepare worker arguments
            worker_args = []
            for worker_id in range(workers):
                args = (
                    width,
                    height,
                    population_rate,
                    toroidal,
                    max_generations,
                    pattern,
                    pattern_x,
                    pattern_y,
                    condition_type,
                    condition_value,
                    seed,
                    worker_id,
                    work_queue,
                    stop_flag,
                    batch_size,
                    progress_counter,
                    progress_lock,
                    continue_search,
                    found_patterns_queue,
                    topology,
                )
                worker_args.append(args)

            # Set up signal handler for graceful interruption
            interrupted = False

            def signal_handler(signum, frame):
                nonlocal interrupted
                interrupted = True
                stop_flag.value = True  # Signal workers to stop
                print("\n⚠️ Search interrupted by user (Ctrl+C)")

            original_handler = signal.signal(signal.SIGINT, signal_handler)

            # Run parallel search
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all workers
                future_to_worker = {
                    executor.submit(_search_worker_batched, args): i for i, args in enumerate(worker_args)
                }

                # Start progress monitoring in a separate thread
                import threading

                progress_thread_stop = threading.Event()
                pattern_save_stop = threading.Event()
                saved_patterns = []  # Track saved patterns
                pattern_count_lock = threading.Lock()
                pattern_counter = 0

                def progress_monitor():
                    last_progress_time = start_time
                    last_count = 0

                    while not progress_thread_stop.is_set() and not stop_flag.value:
                        time.sleep(1)  # Check every second
                        current_time = time.time()

                        # Get current progress
                        with progress_lock:
                            current_count = progress_counter.value

                        # Show progress every 2 seconds
                        if current_time - last_progress_time >= 2.0 and current_count > last_count:
                            elapsed = current_time - start_time
                            attempts_per_sec = current_count / elapsed if elapsed > 0 else 0
                            remaining_attempts = actual_max_attempts - current_count
                            eta_seconds = remaining_attempts / attempts_per_sec if attempts_per_sec > 0 else 0
                            progress_pct = min(100.0, (current_count / actual_max_attempts) * 100)

                            print(
                                f"Progress: {current_count}/{actual_max_attempts} ({progress_pct:.1f}%) - "
                                f"{attempts_per_sec:.1f} attempts/sec - ETA: {eta_seconds:.0f}s"
                            )
                            last_progress_time = current_time
                            last_count = current_count

                def pattern_saver():
                    """Monitor queue and save patterns in real-time."""
                    nonlocal pattern_counter
                    
                    while not pattern_save_stop.is_set():
                        try:
                            # Check for patterns with timeout
                            if not found_patterns_queue.empty():
                                pattern_data = found_patterns_queue.get_nowait()
                                
                                with pattern_count_lock:
                                    pattern_counter += 1
                                    current_count = pattern_counter
                                
                                # Save pattern if requested
                                if save_pattern:
                                    saved_file = self._save_successful_pattern(
                                        pattern_data["initial_grid"],
                                        f"{save_pattern}_{current_count}",
                                        condition_name,
                                        pattern_data["final_gen"],
                                        pattern_data["reason"],
                                        pattern_data["stats"],
                                        pattern_data["attempt"],
                                        width,
                                        height,
                                        population_rate,
                                        toroidal,
                                        max_generations,
                                        pattern,
                                        pattern_x,
                                        pattern_y,
                                        seed,
                                        topology,
                                    )
                                    if saved_file:
                                        # Always print when saving, include relevant details
                                        details = []
                                        # Include cycle length for any cycle-based condition
                                        if pattern_data["reason"] == "cycle" and pattern_data["stats"].get("cycle_length"):
                                            details.append(f"cycle length {pattern_data['stats']['cycle_length']}")
                                        details.append(f"attempt {pattern_data['attempt']}")
                                        print(f"💾 Saved pattern #{current_count}: {saved_file} ({', '.join(details)})")
                                        saved_patterns.append(pattern_data)
                            else:
                                time.sleep(0.1)  # Short sleep when queue is empty
                        except Exception:
                            time.sleep(0.1)  # On error, just wait a bit

                # Start threads
                progress_thread = threading.Thread(target=progress_monitor, daemon=True)
                progress_thread.start()
                
                pattern_saver_thread = None
                if save_pattern and continue_search:
                    pattern_saver_thread = threading.Thread(target=pattern_saver, daemon=True)
                    pattern_saver_thread.start()

                total_attempts = 0
                worker_stats = {}
                combined_end_state_stats = {
                    "cycles": {},
                    "extinctions": 0,
                    "max_generations": 0,
                    "total_attempts": 0,
                    "total_generations": 0,
                }

                try:
                    success_result = None
                    workers_completed = 0

                    for future in as_completed(future_to_worker):
                        (
                            found,
                            worker_id,
                            attempt,
                            final_gen,
                            reason,
                            stats,
                            initial_grid,
                            worker_attempts,
                            worker_end_state_stats,
                        ) = future.result()
                        workers_completed += 1
                        worker_stats[worker_id] = worker_attempts

                        # Always combine end state statistics from this worker
                        combined_end_state_stats["total_attempts"] += worker_end_state_stats["total_attempts"]
                        combined_end_state_stats["extinctions"] += worker_end_state_stats["extinctions"]
                        combined_end_state_stats["max_generations"] += worker_end_state_stats["max_generations"]
                        combined_end_state_stats["total_generations"] += worker_end_state_stats["total_generations"]

                        # Merge cycle statistics
                        for cycle_length, count in worker_end_state_stats["cycles"].items():
                            combined_end_state_stats["cycles"][cycle_length] = (
                                combined_end_state_stats["cycles"].get(cycle_length, 0) + count
                            )

                        if found and success_result is None:
                            # Store the success result but don't return yet - collect remaining worker stats
                            total_attempts = sum(worker_stats.values())

                            # Add search timing information to stats (keep original simulation timing intact)
                            elapsed = time.time() - start_time
                            stats["search_duration_seconds"] = elapsed
                            # Don't overwrite the simulation's own generation rate

                            if verbose:
                                print(f"✓ Found matching configuration! Worker {worker_id}, attempt {attempt}")
                                print(f"Search completed in {elapsed:.2f}s using {workers} workers")

                            # Save pattern if requested
                            saved_file = None
                            if save_pattern and initial_grid:
                                saved_file = self._save_successful_pattern(
                                    initial_grid,
                                    save_pattern,
                                    condition_name,
                                    final_gen,
                                    reason,
                                    stats,
                                    total_attempts,
                                    width,
                                    height,
                                    population_rate,
                                    toroidal,
                                    max_generations,
                                    pattern,
                                    pattern_x,
                                    pattern_y,
                                    seed,
                                    topology,
                                )

                            success_result = (True, attempt, (final_gen, reason, stats, saved_file))

                            # Signal other workers to stop
                            stop_flag.value = True

                            # If we have some workers left, wait a bit for them to finish their current batches
                            if workers_completed < workers and verbose:
                                print(
                                    f"Waiting for remaining {workers - workers_completed} workers "
                                    f"to finish current batches..."
                                )

                    # Calculate final totals after all workers have reported
                    final_total_attempts = sum(worker_stats.values())

                    # Process all found patterns from the queue (for continue_search mode)
                    # In continue mode with save_pattern, patterns are already saved by the real-time thread
                    if continue_search and save_pattern:
                        # Patterns were already saved in real-time, just collect stats
                        found_patterns = saved_patterns  # Use the patterns saved by the thread
                    else:
                        # Process queue for non-continue mode or when not saving
                        found_patterns = []
                        pattern_count = 0
                        # Only process queue if we're in continue mode or don't have a success result yet
                        if continue_search or success_result is None:
                            while not found_patterns_queue.empty():
                                try:
                                    pattern_data = found_patterns_queue.get_nowait()
                                    found_patterns.append(pattern_data)
                                    pattern_count += 1
                                    
                                    # Save each pattern if save_pattern is specified
                                    if save_pattern:
                                        saved_file = self._save_successful_pattern(
                                        pattern_data["initial_grid"],
                                        f"{save_pattern}_{pattern_count}",
                                        condition_name,
                                        pattern_data["final_gen"],
                                        pattern_data["reason"],
                                        pattern_data["stats"],
                                        pattern_data["attempt"],
                                        width,
                                        height,
                                        population_rate,
                                        toroidal,
                                        max_generations,
                                        pattern,
                                        pattern_x,
                                        pattern_y,
                                        seed,
                                        topology,
                                        )
                                        if saved_file:
                                            # Always print when saving, include relevant details
                                            details = []
                                            # Include cycle length for any cycle-based condition
                                            if pattern_data["reason"] == "cycle" and pattern_data["stats"].get("cycle_length"):
                                                details.append(f"cycle length {pattern_data['stats']['cycle_length']}")
                                            details.append(f"attempt {pattern_data['attempt']}")
                                            print(f"💾 Saved pattern #{pattern_count}: {saved_file} ({', '.join(details)})")
                                except:
                                    break

                    if continue_search and found_patterns:
                        # In continue mode, report all found patterns
                        elapsed = time.time() - start_time
                        if verbose:
                            print(f"\n🎯 CONTINUE MODE: Found {len(found_patterns)} matching patterns")
                            print(f"Total attempts made: {final_total_attempts} (across {len(worker_stats)} workers)")
                            print(f"Search completed in {elapsed:.2f}s using {workers} workers")
                            attempts_per_sec = final_total_attempts / elapsed if elapsed > 0 else 0
                            print(f"Search rate: {attempts_per_sec:.1f} attempts/second")
                            
                            for i, pattern_data in enumerate(found_patterns, 1):
                                print(f"Pattern #{i}: Worker {pattern_data['worker_id']}, attempt {pattern_data['attempt']}")
                        
                        # Return the first pattern as the "primary" result, but include all patterns
                        first_pattern = found_patterns[0]
                        first_pattern["stats"]["search_duration_seconds"] = elapsed
                        result_tuple = (first_pattern["final_gen"], first_pattern["reason"], first_pattern["stats"], None)
                        return True, final_total_attempts, result_tuple, combined_end_state_stats

                    # If we found a successful result, return it with complete end state stats
                    if success_result:
                        # Update the success result with final attempt count
                        success_found, match_attempt, result_tuple = success_result
                        final_gen, reason, stats, saved_file = result_tuple

                        if verbose:
                            print(f"Total attempts made: {final_total_attempts} (across {len(worker_stats)} workers)")
                            attempts_per_sec = (
                                final_total_attempts / stats["search_duration_seconds"]
                                if stats["search_duration_seconds"] > 0
                                else 0
                            )
                            print(f"Search rate: {attempts_per_sec:.1f} attempts/second")
                            print(f"Worker distribution: {dict(sorted(worker_stats.items()))}")

                        return True, final_total_attempts, result_tuple, combined_end_state_stats

                    # No worker found a match - collect all worker stats
                    total_attempts = sum(worker_stats.values())
                    elapsed = time.time() - start_time

                    if interrupted:
                        if verbose:
                            print(f"⚠️ Search interrupted after {total_attempts} attempts")
                            print(f"Search completed in {elapsed:.2f}s using {workers} workers")
                            print(f"Batch size: {batch_size}")
                            attempts_per_sec = total_attempts / elapsed if elapsed > 0 else 0
                            print(f"Search rate: {attempts_per_sec:.1f} attempts/second")
                            print(f"Worker distribution: {dict(sorted(worker_stats.items()))}")
                    else:
                        if verbose:
                            print(f"✗ No matching configuration found after {total_attempts} attempts")
                            print(f"Search completed in {elapsed:.2f}s using {workers} workers")
                            print(f"Batch size: {batch_size}, Total batches processed: {total_batches}")
                            attempts_per_sec = total_attempts / elapsed if elapsed > 0 else 0
                            print(f"Search rate: {attempts_per_sec:.1f} attempts/second")
                            print(f"Worker distribution: {dict(sorted(worker_stats.items()))}")

                    return False, total_attempts, {"duration_seconds": elapsed}, combined_end_state_stats

                finally:
                    # Stop progress monitoring
                    progress_thread_stop.set()
                    if progress_thread.is_alive():
                        progress_thread.join(timeout=1)
                    
                    # Stop pattern saver thread
                    if pattern_saver_thread:
                        pattern_save_stop.set()
                        if pattern_saver_thread.is_alive():
                            pattern_saver_thread.join(timeout=1)

                    # Restore original signal handler
                    signal.signal(signal.SIGINT, original_handler)

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
        topology: Optional[str] = None,
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
                    "topology": topology if topology else ("toroidal" if toroidal else "bounded"),
                    "population_rate": population_rate,
                    "max_generations": max_generations,
                    "pattern": pattern,
                    "pattern_x": pattern_x,
                    "pattern_y": pattern_y,
                    "seed": seed,
                },
                "statistics": stats,
                "reload_command": self._generate_reload_command(
                    pattern, width, height, toroidal, population_rate, max_generations, pattern_x, pattern_y, seed, topology
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
        topology: Optional[str] = None,
    ) -> str:
        """Generate a command line to reload this exact simulation configuration."""
        cmd_parts = ["cellular-cli"]

        # Grid parameters
        cmd_parts.append(f"--width {width}")
        cmd_parts.append(f"--height {height}")
        if topology:
            cmd_parts.append(f"--topology {topology}")
        elif toroidal:
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

    def reaches_population_threshold(min_population: int):
        """Condition: reaches at least the specified population at some point."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            # Check if any point in history reached the threshold
            pop_history = stats.get("population_history", [0])
            if not pop_history:
                return False
            max_pop = max(pop_history)
            return max_pop >= min_population

        return condition

    def stabilizes_with_population(target_population: int):
        """Condition: stabilizes (cycles or still life) with specific population."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            if reason in ["cycle", "max_generations"]:
                return stats.get("population", 0) == target_population
            return False

        return condition

    def has_bounding_box_size(min_width: int, min_height: int):
        """Condition: final pattern has bounding box of at least specified size."""

        def condition(final_gen: int, reason: str, stats: Dict[str, Any]) -> bool:
            bbox_size = stats.get("bounding_box_size", (0, 0))
            return bbox_size[0] >= min_width and bbox_size[1] >= min_height

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
        "reaches_population_threshold": reaches_population_threshold,
        "stabilizes_with_population": stabilizes_with_population,
        "has_bounding_box_size": has_bounding_box_size,
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

        elif condition_type == "population_threshold":
            value = int(value_str)
            func = condition_funcs["reaches_population_threshold"](value)
            desc = f"reaches population of at least {value}"

        elif condition_type == "stabilizes_with_population":
            value = int(value_str)
            func = condition_funcs["stabilizes_with_population"](value)
            desc = f"stabilizes with exactly {value} cells"

        elif condition_type == "bounding_box_size":
            if "x" not in value_str:
                raise ValueError("bounding_box_size requires format 'WxH' (e.g., '10x5')")
            width_str, height_str = value_str.split("x")
            width, height = int(width_str), int(height_str)
            func = condition_funcs["has_bounding_box_size"](width, height)
            desc = f"has bounding box of at least {width}x{height}"
            value = (width, height)

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
                "population_threshold",
                "stabilizes_with_population",
                "bounding_box_size",
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
  
  # Run 1000 bulk simulations with metrics collection
  cellular-cli --bulk 1000 --bulk-export results
  
  # Parameter sweep: test different population rates and grid sizes
  cellular-cli --bulk 100 --param-sweep 'population=0.1,0.2,0.3' --param-sweep 'width=50,100' --bulk-export sweep_results

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
        help="Enable toroidal (wraparound) edges (deprecated, use --topology instead)",
    )
    
    parser.add_argument(
        "--topology",
        type=str,
        choices=["toroidal", "bounded", "mirroring"],
        default=None,
        help="Grid edge topology: toroidal (wrap), bounded (dead outside), or mirroring (reflect)",
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
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto-detect CPU count, 1 = sequential)",
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
    
    # Bulk simulation arguments
    parser.add_argument(
        "--bulk",
        type=int,
        metavar="N",
        help="Run N bulk simulations and collect metrics",
    )
    
    parser.add_argument(
        "--bulk-export",
        type=str,
        metavar="FILENAME",
        help="Base filename for bulk simulation results (exports .json, .csv, and _summary.csv)",
    )
    
    parser.add_argument(
        "--param-sweep",
        type=str,
        metavar="PARAM=VALUES",
        action="append",
        help="Parameter sweep for bulk simulations (e.g., --param-sweep 'population=0.1,0.2,0.3' --param-sweep 'width=50,100')",
    )
    
    parser.add_argument(
        "--runs-per-combo",
        type=int,
        default=10,
        help="Number of runs per parameter combination in sweep mode (default: 10)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible simulations",
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
            if not any("--toroidal" in arg or "-t" in arg for arg in sys.argv) and not any("--topology" in arg for arg in sys.argv):
                args.toroidal = sim_params.get("toroidal", args.toroidal)
                # Also check for saved topology
                if "topology" in sim_params and not args.topology:
                    topology = sim_params["topology"]

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

    # Handle topology parameter - determine which topology to use
    if args.topology:
        # New topology parameter takes precedence
        topology = args.topology
        # Update toroidal flag for backward compatibility
        args.toroidal = (topology == "toroidal")
    else:
        # Use deprecated toroidal flag for backward compatibility
        topology = "toroidal" if args.toroidal else "bounded"
    
    try:
        # Handle search mode
        if args.search:
            try:
                condition_func, condition_desc = parse_search_condition(args.search)
                # Parse condition type and value for parallel search
                condition_type, value_str = args.search.split(":", 1)
                if condition_type == "bounding_box_size":
                    width_str, height_str = value_str.split("x")
                    condition_value = (int(width_str), int(height_str))
                elif condition_type == "any_cycle_excluding":
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

            # Determine number of workers
            if args.workers == 0:
                # Auto-detect: use parallel for large searches, sequential for small ones
                workers = mp.cpu_count() if args.search_attempts >= 100 else 1
            else:
                workers = args.workers

            # Start overall timing (includes process startup/shutdown overhead)
            overall_start_time = time.time()

            if workers > 1:
                print(f"Using {workers} workers (batched work stealing)")
            else:
                print(f"Sequential search: {args.search_attempts} attempts")

            # Always use parallel search method (handles workers=1 for sequential)
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
                workers=workers,
                continue_search=args.continue_search,
                topology=topology,
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

        # Handle bulk simulations
        elif args.bulk:
            # Create base configuration
            config = SimulationConfig(
                width=args.width,
                height=args.height,
                toroidal=args.toroidal,
                topology=topology,
                max_generations=args.max_generations,
                population_rate=args.population,
                pattern=args.pattern,
                pattern_x=args.pattern_x,
                pattern_y=args.pattern_y,
                seed=args.seed,
            )
            
            # Create bulk runner
            workers = args.workers if args.workers > 0 else None
            runner = BulkSimulationRunner(
                base_config=config,
                num_runs=args.bulk,
                parallel=(workers != 1),
                workers=workers,
                verbose=args.verbose,
            )
            
            # Handle parameter sweeps if specified
            if args.param_sweep:
                # Parse parameter sweep specifications
                param_ranges = {}
                for sweep_spec in args.param_sweep:
                    if '=' not in sweep_spec:
                        print(f"Error: Invalid parameter sweep format: '{sweep_spec}'")
                        print("Expected format: PARAM=VALUE1,VALUE2,VALUE3")
                        return 1
                    
                    param_name, values_str = sweep_spec.split('=', 1)
                    param_name = param_name.strip()
                    
                    # Parse values based on parameter type
                    try:
                        if param_name in ['width', 'height', 'max_generations']:
                            # Integer parameters
                            values = [int(v.strip()) for v in values_str.split(',')]
                        elif param_name in ['population_rate', 'population']:
                            # Float parameters
                            values = [float(v.strip()) for v in values_str.split(',')]
                        elif param_name in ['toroidal']:
                            # Boolean parameters
                            values = [v.strip().lower() == 'true' for v in values_str.split(',')]
                        elif param_name in ['topology', 'pattern']:
                            # String parameters
                            values = [v.strip() for v in values_str.split(',')]
                        else:
                            print(f"Error: Unknown parameter '{param_name}'")
                            print("Valid parameters: width, height, population_rate, toroidal, topology, max_generations, pattern")
                            return 1
                        
                        # Handle population vs population_rate
                        if param_name == 'population':
                            param_name = 'population_rate'
                        
                        param_ranges[param_name] = values
                        
                    except ValueError as e:
                        print(f"Error parsing values for parameter '{param_name}': {e}")
                        return 1
                
                print(f"\nRunning parameter sweep:")
                for param, values in param_ranges.items():
                    print(f"  {param}: {values}")
                print(f"  Runs per combination: {args.runs_per_combo}")
                print()
                
                # Run parameter sweep
                all_metrics = runner.run_with_parameter_sweep(param_ranges, args.runs_per_combo)
            else:
                # Run standard bulk simulations
                print(f"\nRunning {args.bulk} bulk simulations...")
                all_metrics = runner.run()
            
            # Print summary
            runner.print_summary()
            
            # Export results if requested
            if args.bulk_export:
                output_dir = runner.export_results(args.bulk_export)
                print(f"\nTo visualize results, run:")
                print(f"  python visualize_results.py {output_dir}/{args.bulk_export}.json")
            
            return 0
            
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
                topology=topology,
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
