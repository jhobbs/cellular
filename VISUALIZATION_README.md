# Visualization Guide for Bulk Simulations

This guide explains how to visualize and analyze results from bulk simulations.

## Overview

The visualization system creates comprehensive charts and graphs from bulk simulation results, making it easy to understand patterns and relationships in your data.

## Quick Start

### 1. Run a bulk simulation with export

```bash
# Simple bulk run
cellular-cli --bulk 100 --bulk-export my_experiment

# Parameter sweep
cellular-cli --bulk 100 \
  --param-sweep 'population=0.1,0.2,0.3' \
  --param-sweep 'width=30,50' \
  --runs-per-combo 10 \
  --bulk-export sweep_experiment
```

### 2. Visualize results

```bash
# Visualize using the path shown in CLI output
python visualize_results.py results/my_experiment_20250530_123456/my_experiment.json

# Or specify custom output directory
python visualize_results.py results/sweep_experiment_*/sweep_experiment.json \
  -o results/sweep_experiment_*/visualizations
```

## Directory Structure

Results are organized with timestamps to prevent overwrites:

```
results/
├── experiment_name_20250530_123456/
│   ├── experiment_name.json          # Full simulation data
│   ├── experiment_name.csv           # Tabular format
│   ├── experiment_name_summary.csv   # Aggregate statistics
│   ├── metadata.json                 # Run configuration
│   └── visualizations/               # Generated plots (optional)
│       ├── summary_dashboard.png
│       ├── analysis_report.txt
│       └── ... (other plots)
```

## Generated Visualizations

### 1. Summary Dashboard
A multi-panel overview showing:
- Termination reason pie chart
- Final generation distribution histogram
- Population statistics boxplots
- Cycle length distribution
- Performance metrics scatter plot
- Grid size effects (if applicable)

### 2. Parameter-Specific Plots

For each swept parameter, the system generates:

#### Termination Reason Chart
Shows how different parameter values affect simulation outcomes (cycles, extinction, max generations).

#### Cycle Distribution
Stacked bar chart showing the distribution of cycle lengths for each parameter value.

#### Population Dynamics
Line plots showing population over time for each parameter value, with individual runs shown in light colors and the average in bold.

### 3. Parameter Interaction Heatmaps

When multiple parameters are swept, heatmaps show interactions:
- Average final generation
- Average final population  
- Cycle detection rate

### 4. Scatter Plots

Relationships between metrics:
- Initial vs final population (colored by termination reason)
- Final generation vs average population
- Custom metric comparisons

## Analysis Report

The `analysis_report.txt` provides:
- Summary statistics
- Parameter-specific insights
- Average values with standard deviations
- Termination distributions
- Performance metrics

## Examples

### Example 1: Population Density Study

```bash
# Test different initial densities
cellular-cli --bulk 100 \
  --param-sweep 'population=0.05,0.1,0.15,0.2,0.25,0.3' \
  --runs-per-combo 20 \
  --bulk-export density_study

# Visualize
python visualize_results.py results/density_study_*/density_study.json
```

This creates:
- Population dynamics for each density
- Termination reason trends
- Cycle length distributions

### Example 2: Grid Size Comparison

```bash
# Compare different grid sizes
cellular-cli --bulk 100 \
  --param-sweep 'width=20,40,60,80' \
  --param-sweep 'height=20,40,60,80' \
  --runs-per-combo 5 \
  --bulk-export grid_study

# Visualize with custom output
python visualize_results.py results/grid_study_*/grid_study.json \
  -o grid_analysis
```

This creates:
- Heatmaps showing grid size interactions
- Performance scaling analysis
- Memory usage trends

### Example 3: Pattern Analysis

```bash
# Compare different starting patterns
cellular-cli --bulk 50 \
  --param-sweep 'pattern=Glider,Blinker,Toad,Beacon' \
  --runs-per-combo 10 \
  --bulk-export pattern_comparison

# Visualize
python visualize_results.py results/pattern_comparison_*/pattern_comparison.json
```

## Interpreting Results

### Heatmaps
- **Darker colors** indicate higher values
- **Annotations** show exact values
- Look for **patterns** across parameter combinations
- Identify **sweet spots** for desired behaviors

### Population Dynamics
- **Thin lines**: Individual simulation runs
- **Thick line**: Average across runs
- **Sampling**: Values shown every 100 generations
- **Variability**: Spread indicates randomness effects

### Cycle Distributions
- **Stacked bars**: Show proportion of each cycle length
- **Height**: Total number of runs
- **Colors**: Different cycle lengths
- **Percentages**: Labeled on each segment

### Performance Plots
- **Log scales**: Used for wide value ranges
- **Clustering**: Indicates common performance profiles
- **Outliers**: May indicate edge cases or bugs

## Tips for Effective Analysis

1. **Run enough simulations**: Use at least 10-20 runs per parameter combination for statistical significance

2. **Choose parameter ranges wisely**: Start with wide ranges, then narrow based on initial results

3. **Consider interactions**: Parameters often interact in non-obvious ways - use 2D sweeps to discover these

4. **Save your commands**: The metadata.json file records the exact configuration used

5. **Compare experiments**: Load multiple result sets to compare different approaches

## Customization

The visualization script can be extended:

```python
# Add custom analysis to visualize_results.py
def my_custom_plot(df, output_path):
    # Your visualization code
    plt.savefig(output_path / 'my_custom_plot.png')
```

## Troubleshooting

### Memory Issues
Large parameter sweeps can generate substantial data:
- Reduce `runs_per_combo`
- Process results in batches
- Use sampling for very long simulations

### Missing Visualizations
Some plots are only generated when relevant:
- Cycle plots require simulations that found cycles
- Heatmaps require multiple swept parameters
- Population dynamics require non-empty population histories

### Performance
- Use `--workers` flag to parallelize simulations
- Visualization is single-threaded but optimized
- Consider using a machine with more RAM for large datasets