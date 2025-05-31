#!/usr/bin/env python3
"""Example usage of the bulk simulation metrics framework."""

from cellular.core.grid import Grid
from cellular.core.game import GameOfLife
from cellular.core.bulk_runner import BulkSimulationRunner, SimulationConfig
from cellular.core.metrics import MetricsCollector, MetricsAggregator, MetricsExporter


def main():
    """Demonstrate bulk simulation capabilities."""
    print("=== Cellular Automata Bulk Simulation Example ===\n")
    
    # Example 1: Basic bulk simulation with random initial conditions
    print("Example 1: Running 100 simulations with random initial conditions")
    print("-" * 60)
    
    config = SimulationConfig(
        width=50,
        height=50,
        toroidal=True,
        population_rate=0.3,
        max_generations=1000
    )
    
    runner = BulkSimulationRunner(
        base_config=config,
        num_runs=100,
        parallel=True,
        verbose=True
    )
    
    metrics = runner.run()
    runner.print_summary()
    runner.export_results("example1_random_simulations")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: Parameter sweep
    print("Example 2: Parameter sweep across population densities")
    print("-" * 60)
    
    config2 = SimulationConfig(
        width=30,
        height=30,
        toroidal=True,
        max_generations=500
    )
    
    runner2 = BulkSimulationRunner(
        base_config=config2,
        num_runs=0,  # Not used in parameter sweep
        parallel=True,
        verbose=True
    )
    
    # Test different population densities
    param_ranges = {
        'population_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    
    metrics2 = runner2.run_with_parameter_sweep(param_ranges, runs_per_combination=20)
    runner2.print_summary()
    runner2.export_results("example2_population_sweep")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 3: Custom metrics collection with hooks
    print("Example 3: Custom metrics with density tracking")
    print("-" * 60)
    
    # Define a custom metric hook
    def track_density_changes(metrics, game):
        """Custom hook to track density changes."""
        current_density = game.population / (game.grid.width * game.grid.height)
        
        if 'density_history' not in metrics.custom_metrics:
            metrics.custom_metrics['density_history'] = []
        
        metrics.custom_metrics['density_history'].append(current_density)
        
        # Track if density ever drops below 10%
        if current_density < 0.1:
            metrics.custom_metrics['low_density_reached'] = True
    
    # Create a single simulation with custom metrics
    grid = Grid(40, 40, wrap_edges=True)
    grid.randomize(0.35)
    game = GameOfLife(grid)
    
    collector = MetricsCollector()
    collector.register_hook('update', track_density_changes)
    
    # Run simulation with metrics collection
    collector.start_run(1, game)
    
    for _ in range(500):
        game.step()
        if game.generation % 10 == 0:  # Update metrics every 10 generations
            collector.update(game)
        
        if game.cycle_detected or game.population == 0:
            break
    
    # Finalize metrics
    reason = "cycle" if game.cycle_detected else ("extinction" if game.population == 0 else "max_generations")
    final_metrics = collector.end_run(game, game.generation, reason)
    
    print(f"Simulation completed: {reason} at generation {game.generation}")
    print(f"Low density reached: {final_metrics.custom_metrics.get('low_density_reached', False)}")
    print(f"Final density: {game.population / (game.grid.width * game.grid.height):.2%}")
    
    # Export single run
    MetricsExporter.to_json([final_metrics], "example3_custom_metrics.json")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 4: Analyzing specific patterns
    print("Example 4: Bulk analysis of specific starting patterns")
    print("-" * 60)
    
    # Test multiple runs of the same pattern to see variability
    config4 = SimulationConfig(
        width=100,
        height=100,
        toroidal=False,
        pattern="R-pentomino",
        pattern_x=45,
        pattern_y=45,
        max_generations=2000
    )
    
    runner4 = BulkSimulationRunner(
        base_config=config4,
        num_runs=10,  # Run the same pattern 10 times
        parallel=True,
        verbose=True
    )
    
    metrics4 = runner4.run()
    runner4.print_summary()
    
    # Since it's the same deterministic pattern, all runs should be identical
    print("\nNote: All runs should produce identical results for deterministic patterns")
    
    print("\n=== Examples Complete ===")
    print("\nGenerated output files:")
    print("  - example1_random_simulations.json/.csv")
    print("  - example2_population_sweep.json/.csv")  
    print("  - example3_custom_metrics.json")
    print("\nUse these files for further analysis or visualization!")


if __name__ == "__main__":
    main()