"""Bulk simulation runner for Game of Life with metrics collection."""

import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass
import time
import signal
import sys

from .grid import Grid
from .game import GameOfLife
from .patterns import PatternLibrary, Pattern
from .metrics import MetricsCollector, MetricsAggregator, MetricsExporter, SimulationMetrics, NumpyEncoder


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    width: int = 50
    height: int = 50
    toroidal: bool = True
    topology: str = "moore8"
    max_generations: int = 10000
    population_rate: float = 0.3
    pattern: Optional[str] = None
    pattern_x: Optional[int] = None
    pattern_y: Optional[int] = None
    seed: Optional[int] = None


class BulkSimulationRunner:
    """Runner for bulk Game of Life simulations with metrics collection."""
    
    def __init__(self, 
                 base_config: SimulationConfig,
                 num_runs: int = 100,
                 parallel: bool = True,
                 workers: Optional[int] = None,
                 verbose: bool = True):
        """Initialize bulk simulation runner.
        
        Args:
            base_config: Base configuration for simulations
            num_runs: Number of simulation runs to perform
            parallel: Whether to run simulations in parallel
            workers: Number of worker processes (None for CPU count)
            verbose: Whether to print progress updates
        """
        self.base_config = base_config
        self.num_runs = num_runs
        self.parallel = parallel
        self.workers = workers or mp.cpu_count()
        self.verbose = verbose
        
        self.aggregator = MetricsAggregator()
        self.interrupted = False
        
        # Setup interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle keyboard interrupt gracefully."""
        self.interrupted = True
        if self.verbose:
            print("\n\nInterrupted! Finishing current runs and saving results...")
    
    def run_single_simulation(self, run_id: int, config: SimulationConfig) -> SimulationMetrics:
        """Run a single simulation with metrics collection."""
        # Set random seed if specified
        if config.seed is not None:
            random.seed(config.seed + run_id)
        
        # Create grid and game
        grid = Grid(config.width, config.height, 
                   wrap_edges=config.toroidal, 
                   topology=config.topology)
        game = GameOfLife(grid)
        
        # Initialize pattern or random state
        pattern_library = PatternLibrary()
        initial_pattern_name = None
        
        if config.pattern:
            pattern = pattern_library.get_pattern(config.pattern)
            if pattern:
                pattern.apply_to_grid(grid, config.pattern_x, config.pattern_y)
                initial_pattern_name = config.pattern
            else:
                grid.randomize(config.population_rate)
        else:
            grid.randomize(config.population_rate)
        
        # Create metrics collector
        collector = MetricsCollector()
        collector.start_run(run_id, game, initial_pattern_name)
        
        # Run simulation with periodic metrics updates
        update_interval = 100  # Update metrics every 100 generations
        last_update = 0
        
        while game.generation < config.max_generations:
            game.step()
            
            # Update metrics periodically
            if game.generation - last_update >= update_interval:
                collector.update(game)
                last_update = game.generation
            
            # Check termination conditions
            if game.cycle_detected or game.population == 0:
                break
        
        # Final metrics update
        collector.update(game)
        
        # Determine termination reason
        if game.cycle_detected:
            reason = "cycle"
        elif game.population == 0:
            reason = "extinction"
        else:
            reason = "max_generations"
        
        # Finalize metrics
        metrics = collector.end_run(game, game.generation, reason)
        return metrics
    
    def run_with_parameter_sweep(self, 
                                parameter_ranges: Dict[str, List[Any]],
                                runs_per_combination: int = 10) -> List[SimulationMetrics]:
        """Run simulations with parameter sweeps.
        
        Args:
            parameter_ranges: Dictionary of parameter names to lists of values
            runs_per_combination: Number of runs per parameter combination
            
        Returns:
            List of all simulation metrics
        """
        # Generate all parameter combinations
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        total_runs = len(combinations) * runs_per_combination
        if self.verbose:
            print(f"Running {total_runs} simulations ({len(combinations)} parameter combinations)")
        
        all_metrics = []
        run_id = 0
        
        for combo in combinations:
            if self.interrupted:
                break
                
            # Create config for this combination
            config = SimulationConfig(**vars(self.base_config))
            for i, param_name in enumerate(param_names):
                setattr(config, param_name, combo[i])
            
            # Run multiple simulations for this combination
            for _ in range(runs_per_combination):
                if self.interrupted:
                    break
                    
                metrics = self.run_single_simulation(run_id, config)
                all_metrics.append(metrics)
                self.aggregator.add_run(metrics)
                run_id += 1
                
                if self.verbose and run_id % 10 == 0:
                    print(f"Completed {run_id}/{total_runs} runs...")
        
        return all_metrics
    
    def run(self) -> List[SimulationMetrics]:
        """Run bulk simulations.
        
        Returns:
            List of metrics for all runs
        """
        if self.parallel and self.num_runs > 1:
            return self._run_parallel()
        else:
            return self._run_sequential()
    
    def _run_sequential(self) -> List[SimulationMetrics]:
        """Run simulations sequentially."""
        all_metrics = []
        
        for run_id in range(self.num_runs):
            if self.interrupted:
                break
                
            if self.verbose and run_id % 10 == 0:
                print(f"Running simulation {run_id + 1}/{self.num_runs}...")
            
            metrics = self.run_single_simulation(run_id, self.base_config)
            all_metrics.append(metrics)
            self.aggregator.add_run(metrics)
        
        return all_metrics
    
    def _run_parallel(self) -> List[SimulationMetrics]:
        """Run simulations in parallel."""
        all_metrics = []
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            futures = {}
            for run_id in range(self.num_runs):
                if self.interrupted:
                    break
                    
                future = executor.submit(
                    _run_single_simulation_worker,
                    run_id, self.base_config
                )
                futures[future] = run_id
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                if self.interrupted:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                    
                try:
                    metrics = future.result()
                    all_metrics.append(metrics)
                    self.aggregator.add_run(metrics)
                    completed += 1
                    
                    if self.verbose and completed % 10 == 0:
                        print(f"Completed {completed}/{self.num_runs} runs...")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error in run {futures[future]}: {e}")
        
        return all_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all runs."""
        return self.aggregator.get_summary_statistics()
    
    def export_results(self, base_filename: str = "simulation_results", 
                      results_dir: str = "results"):
        """Export results to JSON and CSV formats in organized directory.
        
        Args:
            base_filename: Base name for output files
            results_dir: Directory to store results (created if doesn't exist)
        """
        from pathlib import Path
        from datetime import datetime
        
        # Create results directory if it doesn't exist
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        # Create timestamp subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_path / f"{base_filename}_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Get all metrics
        all_metrics = list(self.aggregator.runs)
        
        # Export detailed results
        json_path = run_dir / f"{base_filename}.json"
        csv_path = run_dir / f"{base_filename}.csv"
        summary_path = run_dir / f"{base_filename}_summary.csv"
        
        MetricsExporter.to_json(all_metrics, str(json_path))
        MetricsExporter.to_csv(all_metrics, str(csv_path))
        MetricsExporter.to_summary_csv(self.aggregator, str(summary_path))
        
        # Create a metadata file with run information
        metadata = {
            "timestamp": timestamp,
            "base_config": vars(self.base_config),
            "num_runs": self.num_runs,
            "parallel": self.parallel,
            "workers": self.workers,
            "total_runs_completed": len(all_metrics),
            "summary": self.aggregator.get_summary_statistics()
        }
        
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        
        if self.verbose:
            print(f"\nResults exported to: {run_dir}/")
            print(f"  - {base_filename}.json (full data)")
            print(f"  - {base_filename}.csv (tabular data)")
            print(f"  - {base_filename}_summary.csv (aggregate stats)")
            print(f"  - metadata.json (run configuration)")
            
        return str(run_dir)
    
    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.get_summary()
        
        print("\n=== Simulation Summary ===")
        print(f"Total runs: {summary['total_runs']}")
        
        print("\nTermination reasons:")
        for reason, count in summary['termination_reasons'].items():
            percentage = count / summary['total_runs'] * 100
            print(f"  {reason}: {count} ({percentage:.1f}%)")
        
        print("\nCycle length distribution:")
        cycle_dist = summary['cycle_length_distribution']
        if cycle_dist:
            for length in sorted(cycle_dist.keys()):
                count = cycle_dist[length]
                percentage = count / summary['total_runs'] * 100
                print(f"  Length {length}: {count} ({percentage:.1f}%)")
        
        print(f"\nAverage generations: {summary['generation_stats']['mean']:.1f}")
        print(f"Average duration: {summary['duration_stats']['mean']:.2f}s")
        print(f"Average performance: {summary['performance_stats']['mean_gps']:.1f} gen/s")


def _run_single_simulation_worker(run_id: int, config: SimulationConfig) -> SimulationMetrics:
    """Worker function for parallel simulation execution."""
    runner = BulkSimulationRunner(config, num_runs=1, parallel=False, verbose=False)
    return runner.run_single_simulation(run_id, config)