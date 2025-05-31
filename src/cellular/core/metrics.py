"""Metrics collection framework for bulk simulations."""

import time
import json
import numpy as np
import csv
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from pathlib import Path


@dataclass
class SimulationMetrics:
    """Metrics for a single simulation run."""
    
    # Run identification
    run_id: int
    start_time: float
    end_time: float
    duration: float
    
    # Initial conditions
    initial_population: int
    initial_pattern: Optional[str] = None
    grid_size: Tuple[int, int] = (0, 0)
    toroidal: bool = False
    topology: str = "moore8"
    
    # Simulation outcome
    final_generation: int = 0
    termination_reason: str = ""  # 'cycle', 'extinction', 'max_generations'
    final_population: int = 0
    
    # Cycle information
    cycle_detected: bool = False
    cycle_length: int = 0
    cycle_start_generation: int = 0
    
    # Population dynamics
    population_history: List[int] = field(default_factory=list)
    min_population: int = 0
    max_population: int = 0
    avg_population: float = 0.0
    population_std_dev: float = 0.0
    
    # Spatial metrics
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    bounding_box_size: Tuple[int, int] = (0, 0)
    bounding_box_area: int = 0
    max_bounding_box_area: int = 0
    
    # Performance metrics
    generations_per_second: float = 0.0
    total_cell_updates: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from collected data."""
        if self.population_history:
            self.min_population = min(self.population_history)
            self.max_population = max(self.population_history)
            self.avg_population = np.mean(self.population_history)
            self.population_std_dev = np.std(self.population_history)


class MetricsCollector:
    """Collects metrics during Game of Life simulations."""
    
    def __init__(self):
        self.current_metrics: Optional[SimulationMetrics] = None
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)
        
    def start_run(self, run_id: int, game, initial_pattern: Optional[str] = None):
        """Start collecting metrics for a new run."""
        self.current_metrics = SimulationMetrics(
            run_id=run_id,
            start_time=time.time(),
            end_time=0,
            duration=0,
            initial_population=game.population,
            initial_pattern=initial_pattern,
            grid_size=(game.grid.width, game.grid.height),
            toroidal=game.grid.wrap_edges,
            topology=game.grid.topology,
        )
        
        # Call start hooks
        for hook in self.hooks['start']:
            hook(self.current_metrics, game)
    
    def update(self, game):
        """Update metrics with current game state."""
        if not self.current_metrics:
            return
            
        # Update population history
        self.current_metrics.population_history.append(game.population)
        
        # Update spatial metrics
        bbox = game.grid.get_bounding_box()
        if bbox:
            self.current_metrics.bounding_box = bbox
            width = bbox[2] - bbox[0] + 1
            height = bbox[3] - bbox[1] + 1
            self.current_metrics.bounding_box_size = (width, height)
            area = width * height
            self.current_metrics.bounding_box_area = area
            self.current_metrics.max_bounding_box_area = max(
                self.current_metrics.max_bounding_box_area, area
            )
        
        # Call update hooks
        for hook in self.hooks['update']:
            hook(self.current_metrics, game)
    
    def end_run(self, game, final_generation: int, termination_reason: str):
        """Finalize metrics for the current run."""
        if not self.current_metrics:
            return None
            
        self.current_metrics.end_time = time.time()
        self.current_metrics.duration = self.current_metrics.end_time - self.current_metrics.start_time
        self.current_metrics.final_generation = final_generation
        self.current_metrics.termination_reason = termination_reason
        self.current_metrics.final_population = game.population
        
        # Cycle information
        self.current_metrics.cycle_detected = game.cycle_detected
        self.current_metrics.cycle_length = game.cycle_length
        self.current_metrics.cycle_start_generation = game.cycle_start_generation
        
        # Performance metrics
        if self.current_metrics.duration > 0:
            self.current_metrics.generations_per_second = (
                final_generation / self.current_metrics.duration
            )
        
        self.current_metrics.total_cell_updates = (
            final_generation * game.grid.width * game.grid.height
        )
        
        # Calculate derived metrics
        self.current_metrics.calculate_derived_metrics()
        
        # Call end hooks
        for hook in self.hooks['end']:
            hook(self.current_metrics, game)
        
        return self.current_metrics
    
    def register_hook(self, event: str, hook: Callable):
        """Register a custom hook for metrics collection.
        
        Args:
            event: 'start', 'update', or 'end'
            hook: Callable that takes (metrics, game) as arguments
        """
        if event in ['start', 'update', 'end']:
            self.hooks[event].append(hook)


class MetricsAggregator:
    """Aggregates metrics across multiple simulation runs."""
    
    def __init__(self):
        self.runs: List[SimulationMetrics] = []
        
    def add_run(self, metrics: SimulationMetrics):
        """Add metrics from a single run."""
        self.runs.append(metrics)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all runs."""
        if not self.runs:
            return {}
            
        # Group by termination reason
        termination_counts = defaultdict(int)
        cycle_length_distribution = defaultdict(int)
        
        # Collect aggregate data
        all_durations = []
        all_final_generations = []
        all_final_populations = []
        all_initial_populations = []
        all_max_populations = []
        all_avg_populations = []
        all_gps = []  # generations per second
        
        for run in self.runs:
            termination_counts[run.termination_reason] += 1
            
            if run.cycle_detected:
                cycle_length_distribution[run.cycle_length] += 1
            
            all_durations.append(run.duration)
            all_final_generations.append(run.final_generation)
            all_final_populations.append(run.final_population)
            all_initial_populations.append(run.initial_population)
            all_max_populations.append(run.max_population)
            all_avg_populations.append(run.avg_population)
            all_gps.append(run.generations_per_second)
        
        summary = {
            "total_runs": len(self.runs),
            "termination_reasons": dict(termination_counts),
            "cycle_length_distribution": dict(cycle_length_distribution),
            
            "duration_stats": {
                "mean": np.mean(all_durations),
                "std": np.std(all_durations),
                "min": np.min(all_durations),
                "max": np.max(all_durations),
                "total": np.sum(all_durations),
            },
            
            "generation_stats": {
                "mean": np.mean(all_final_generations),
                "std": np.std(all_final_generations),
                "min": np.min(all_final_generations),
                "max": np.max(all_final_generations),
                "total": np.sum(all_final_generations),
            },
            
            "population_stats": {
                "initial": {
                    "mean": np.mean(all_initial_populations),
                    "std": np.std(all_initial_populations),
                },
                "final": {
                    "mean": np.mean(all_final_populations),
                    "std": np.std(all_final_populations),
                },
                "max_reached": {
                    "mean": np.mean(all_max_populations),
                    "std": np.std(all_max_populations),
                },
                "average": {
                    "mean": np.mean(all_avg_populations),
                    "std": np.std(all_avg_populations),
                },
            },
            
            "performance_stats": {
                "mean_gps": np.mean(all_gps),
                "total_cell_updates": sum(run.total_cell_updates for run in self.runs),
            },
            
            "timestamp": datetime.now().isoformat(),
        }
        
        return summary
    
    def filter_runs(self, filter_func: Callable[[SimulationMetrics], bool]) -> List[SimulationMetrics]:
        """Filter runs based on a condition."""
        return [run for run in self.runs if filter_func(run)]
    
    def group_by(self, key_func: Callable[[SimulationMetrics], Any]) -> Dict[Any, List[SimulationMetrics]]:
        """Group runs by a key function."""
        groups = defaultdict(list)
        for run in self.runs:
            key = key_func(run)
            groups[key].append(run)
        return dict(groups)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class MetricsExporter:
    """Export metrics to various formats."""
    
    @staticmethod
    def to_json(metrics: List[SimulationMetrics], filepath: str, include_summary: bool = True):
        """Export metrics to JSON format."""
        data = {
            "runs": [m.to_dict() for m in metrics],
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_runs": len(metrics),
            }
        }
        
        if include_summary:
            aggregator = MetricsAggregator()
            for m in metrics:
                aggregator.add_run(m)
            data["summary"] = aggregator.get_summary_statistics()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    @staticmethod
    def to_csv(metrics: List[SimulationMetrics], filepath: str):
        """Export metrics to CSV format."""
        if not metrics:
            return
            
        # Flatten metrics for CSV export
        rows = []
        for m in metrics:
            row = {
                'run_id': m.run_id,
                'duration': m.duration,
                'initial_population': m.initial_population,
                'final_population': m.final_population,
                'final_generation': m.final_generation,
                'termination_reason': m.termination_reason,
                'cycle_detected': m.cycle_detected,
                'cycle_length': m.cycle_length,
                'cycle_start_generation': m.cycle_start_generation,
                'min_population': m.min_population,
                'max_population': m.max_population,
                'avg_population': m.avg_population,
                'population_std_dev': m.population_std_dev,
                'bounding_box_area': m.bounding_box_area,
                'max_bounding_box_area': m.max_bounding_box_area,
                'generations_per_second': m.generations_per_second,
                'grid_width': m.grid_size[0],
                'grid_height': m.grid_size[1],
                'toroidal': m.toroidal,
                'initial_pattern': m.initial_pattern or 'random',
            }
            rows.append(row)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    @staticmethod
    def to_summary_csv(aggregator: MetricsAggregator, filepath: str):
        """Export summary statistics to CSV format."""
        summary = aggregator.get_summary_statistics()
        
        # Flatten summary for CSV
        rows = []
        
        # Termination reasons
        for reason, count in summary['termination_reasons'].items():
            rows.append({
                'metric_type': 'termination_reason',
                'metric_name': reason,
                'count': count,
                'percentage': count / summary['total_runs'] * 100,
            })
        
        # Cycle length distribution
        for length, count in summary['cycle_length_distribution'].items():
            rows.append({
                'metric_type': 'cycle_length',
                'metric_name': f'cycle_{length}',
                'count': count,
                'percentage': count / summary['total_runs'] * 100,
            })
        
        with open(filepath, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)