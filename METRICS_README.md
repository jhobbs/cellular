# Bulk Simulation Metrics Documentation

This document explains the metrics collected by the bulk simulation framework and how to interpret them.

## Overview

The bulk simulation framework collects comprehensive metrics from Conway's Game of Life simulations. Metrics are collected at both individual run and aggregate levels.

## Individual Run Metrics

Each simulation run produces a `SimulationMetrics` object containing:

### Run Identification
- **run_id**: Unique identifier for the simulation run
- **start_time**: Unix timestamp when simulation started
- **end_time**: Unix timestamp when simulation ended
- **duration**: Total runtime in seconds

### Initial Conditions
- **initial_population**: Number of live cells at generation 0
- **initial_pattern**: Name of starting pattern (if any, otherwise null for random)
- **grid_size**: Tuple of (width, height) for the simulation grid
- **toroidal**: Boolean indicating if edges wrap around
- **topology**: Grid topology type ("moore8", "toroidal", "bounded")

### Simulation Outcome
- **final_generation**: Generation number when simulation terminated
- **termination_reason**: How the simulation ended:
  - `"cycle"`: Pattern entered a repeating cycle
  - `"extinction"`: All cells died
  - `"max_generations"`: Hit the generation limit
- **final_population**: Number of live cells at termination

### Cycle Information
- **cycle_detected**: Boolean indicating if a cycle was found
- **cycle_length**: Number of generations in the cycle (0 if no cycle)
- **cycle_start_generation**: Generation where the cycle began (0 if no cycle)

### Population Dynamics
- **population_history**: List of population counts sampled periodically (every 100 generations + final)
  - NOT every generation - this is a sampling to reduce memory usage
  - For a 500-generation run, you'd get ~6 values: [initial, gen100, gen200, gen300, gen400, gen500]
- **min_population**: Lowest population in the sampled history (may miss actual minimum)
- **max_population**: Highest population in the sampled history (may miss actual maximum)
- **avg_population**: Average of the sampled population values (not true average across all generations)
- **population_std_dev**: Standard deviation of the sampled population values

### Spatial Metrics
- **bounding_box**: Tuple of (min_x, min_y, max_x, max_y) containing all live cells
- **bounding_box_size**: Tuple of (width, height) of the bounding box
- **bounding_box_area**: Area of the bounding box (width × height)
- **max_bounding_box_area**: Largest bounding box area reached during simulation

### Performance Metrics
- **generations_per_second**: Simulation speed (generations/second)
- **total_cell_updates**: Total number of cell updates (generations × grid_size)

## Aggregate Statistics

When analyzing multiple runs, the framework provides summary statistics:

### Termination Distribution
Shows how simulations ended:
```
termination_reasons: {
  "cycle": 450,        # 45% ended in cycles
  "extinction": 50,    # 5% died out
  "max_generations": 500  # 50% hit generation limit
}
```

### Cycle Length Distribution
For simulations that ended in cycles:
```
cycle_length_distribution: {
  1: 200,  # 200 runs ended in still life (cycle length 1)
  2: 150,  # 150 runs ended in period-2 oscillators
  3: 50,   # 50 runs ended in period-3 oscillators
  ...
}
```

### Duration Statistics
Performance metrics across all runs:
- **mean**: Average simulation duration
- **std**: Standard deviation of durations
- **min/max**: Fastest and slowest runs
- **total**: Total computation time

### Generation Statistics
- **mean**: Average number of generations per run
- **std**: Variation in generation counts
- **min/max**: Shortest and longest runs
- **total**: Total generations computed across all runs

### Population Statistics
Population metrics with subcategories:
- **initial**: Statistics on starting populations
- **final**: Statistics on ending populations
- **max_reached**: Statistics on peak populations
- **average**: Statistics on mean populations per run

## Export Formats

### JSON Export (`filename.json`)
Complete data with:
- Full metrics for each run
- Summary statistics
- Export metadata (timestamp, total runs)

### CSV Export (`filename.csv`)
Flattened metrics, one row per run:
- All scalar metrics as columns
- Boolean values as 0/1
- Useful for analysis in Excel, R, or pandas

### Summary CSV (`filename_summary.csv`)
Aggregate statistics only:
- Termination reason counts and percentages
- Cycle length distribution
- Useful for quick overview

## Usage Examples

### Basic Bulk Simulation
```bash
# Run 1000 simulations, export results
cellular-cli --bulk 1000 --bulk-export experiment1

# Outputs:
# - experiment1.json (full data)
# - experiment1.csv (run-level data)
# - experiment1_summary.csv (aggregate stats)
```

### Parameter Sweep
```bash
# Test different population densities
cellular-cli --bulk 100 \
  --param-sweep 'population=0.1,0.2,0.3,0.4,0.5' \
  --runs-per-combo 20 \
  --bulk-export density_sweep
```

### Analyzing Results in Python
```python
import json
import pandas as pd

# Load full results
with open('experiment1.json', 'r') as f:
    data = json.load(f)

# Summary statistics
print(data['summary'])

# Individual runs
runs_df = pd.DataFrame([run for run in data['runs']])

# Filter to cycles only
cycles_df = runs_df[runs_df['termination_reason'] == 'cycle']

# Analyze cycle lengths
cycle_distribution = cycles_df['cycle_length'].value_counts()
```

## Interpreting Results

### Common Patterns

1. **High extinction rate**: Initial density too low or grid too small
2. **Many max_generations**: Patterns taking long to stabilize, consider increasing limit
3. **Predominant cycle length 1 or 2**: Most patterns quickly reach still lifes or simple oscillators
4. **Large population std_dev**: High variability, possibly chaotic dynamics

### Performance Considerations

- **generations_per_second**: Varies with grid size and population density
- Sparse grids (low population) typically run faster
- Larger grids have more cell updates but may have better vectorization

### Statistical Significance

When comparing parameter sets:
- Run enough simulations for statistical significance (typically 100+)
- Check standard deviations to understand variability
- Use parameter sweeps to identify trends

## Custom Metrics

You can add custom metrics using hooks:

```python
def track_gliders(metrics, game):
    # Custom analysis code
    metrics.custom_metrics['glider_count'] = count_gliders(game.grid)

collector = MetricsCollector()
collector.register_hook('update', track_gliders)
```

Custom metrics appear in the `custom_metrics` dictionary in exports.

## Important Notes on Sampling

### Population History Sampling
The population history is **sampled** every 100 generations, not recorded every generation. This is done to reduce memory usage in bulk simulations. If you need full population history:

1. **Modify the update interval** in `bulk_runner.py`:
   ```python
   update_interval = 1  # Update every generation instead of every 100
   ```

2. **Use a custom hook** for specific metrics:
   ```python
   def track_full_population(metrics, game):
       if 'full_population_history' not in metrics.custom_metrics:
           metrics.custom_metrics['full_population_history'] = []
       metrics.custom_metrics['full_population_history'].append(game.population)
   
   collector.register_hook('update', track_full_population)
   ```

3. **Be aware of memory implications**: Full history for 1000 simulations × 10,000 generations = 10 million data points

### Implications for Analysis
- The min/max population values are based on samples and may miss peaks/valleys
- The average population is an average of samples, not the true time-averaged population
- For accurate population dynamics, consider using smaller update intervals or custom collection