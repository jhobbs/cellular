# TODO

## Research Ideas

- [ ] Explore cycle perturbation dynamics
  - After hitting a cycle, add random cells and measure:
    - How much longer until we hit a cycle again
    - Whether it's the same cycle or a different one
    - How the perturbation size affects recovery time
    - Statistical distribution of recovery times
    - Relationship between cycle length and stability under perturbation
  - Could lead to insights about pattern robustness and basin of attraction

## Feature Enhancements

- [ ] Connected component analysis and visualization
  - Add ability to identify and track connected components
  - Visual features:
    - Draw boundaries around each connected component
    - Color-code different components
    - Optional labeling with component IDs
  - Statistical tracking:
    - Number of distinct connected components over time
    - Size of largest component
    - Average component size
    - Component birth/death events
    - Component merging/splitting detection
  - Implementation considerations:
    - Use scipy.ndimage.label for component identification
    - Efficient tracking across generations (component ID persistence)
    - Handle both GUI visualization and bulk simulation statistics
    - Add to pattern search conditions (e.g., "find patterns that split into N components")

- [x] Bulk simulation metrics collection framework (COMPLETED)
  - Two collection strategies:
    1. **Full granularity mode**:
       - Collect all metrics from every simulation run
       - Store complete time series for each metric
       - Memory-intensive but provides complete data
       - Useful for detailed analysis and outlier detection
    2. **Statistical sampling mode**:
       - Sample subset of runs (stratified or random sampling)
       - Compute confidence intervals and bounds
       - Memory-efficient for large-scale experiments
       - Provides statistical guarantees on population metrics
  - Metrics to track:
    - Population over time
    - Connected component statistics
    - Cycle detection points
    - Bounding box evolution
    - Pattern velocity (for moving patterns)
    - Entropy/complexity measures
  - Output formats:
    - HDF5 for efficient storage of time series data
    - Summary statistics in JSON/CSV
    - Configurable metric aggregation (mean, median, percentiles)
  - Implementation:
    - Metric collector interface for extensibility
    - Efficient circular buffers for streaming collection
    - Real-time statistics computation
    - Integration with existing batch simulation infrastructure

- [ ] Configuration space optimization framework
  - Search for parameter sets that optimize metrics
  - Optimization objectives:
    - Maximize single metrics (e.g., longest cycle length, most components)
    - Minimize cost functions (e.g., time to stability, pattern size)
    - Optimize weighted combinations of metrics
  - Search parameters:
    - Initial population density
    - Grid dimensions
    - Topology type
    - Pattern placement/orientation
    - Perturbation strategies
  - Optimization algorithms:
    - Grid search for small parameter spaces
    - Bayesian optimization for efficient exploration
    - Genetic algorithms for complex objectives
    - Simulated annealing for local refinement
  - Cost/benefit function features:
    - User-defined metric weights
    - Multi-objective optimization (Pareto frontiers)
    - Constraint handling (e.g., max grid size, min population)
    - Time-based penalties (prefer faster convergence)
  - Implementation:
    - Plugin architecture for custom objective functions
    - Integration with metrics collection framework
    - Parallel evaluation of parameter sets
    - Visualization of optimization progress
    - Export of optimal configurations
  - Example use cases:
    - Find initial conditions that produce longest-lived methuselahs
    - Discover parameters yielding maximum pattern diversity
    - Optimize for patterns with specific component structures
    - Search for configurations with rare behaviors

- [ ] Support for alternative cellular automaton rules
  - Start with rules using same 8-neighbor Moore neighborhood
  - Rule specification formats:
    - B/S notation (e.g., B3/S23 for Conway's Life)
    - Numeric rule codes
    - Custom rule functions
  - Initial rule variants to implement:
    - HighLife (B36/S23) - includes replicator
    - Seeds (B2/S) - exploding growth
    - Life Without Death (B3/S012345678) - ink blot growth
    - Day & Night (B3678/S34678) - symmetric rules
    - Morley (B368/S245) - forms intricate patterns
    - 2x2 (B36/S125) - forms block patterns
  - Implementation approach:
    - Rule class/interface for extensibility
    - Efficient rule application using lookup tables
    - Update Grid/Game classes to accept rule parameter
    - Maintain Conway's Life as default
  - UI integration:
    - Rule selection dropdown in GUI
    - --rule parameter in CLI
    - Rule specification in pattern files
  - Extensions:
    - Support for larger neighborhoods (e.g., Moore radius 2)
    - Totalistic vs non-totalistic rules
    - Multi-state automata (e.g., Generations, Brian's Brain)
    - Rule mutation/evolution during simulation
  - Pattern library considerations:
    - Tag patterns with compatible rules
    - Auto-convert patterns between similar rules
    - Rule-specific pattern collections

- [ ] Support for alternative cellular topologies
  - Move beyond square grid to other tessellations and structures
  - Topology types to implement:
    - **Hexagonal grid** - 6 neighbors per cell
      - More natural for some patterns
      - Different neighbor counting rules
      - Requires coordinate system adaptation (axial/cubic)
    - **Triangular grid** - 3 or 12 neighbors depending on orientation
      - Alternating up/down triangles
      - Unique propagation patterns
    - **Penrose tiling** - Aperiodic tiling
      - Non-repeating patterns
      - Variable neighbor counts
    - **3D cubic grid** - 26 neighbors (3x3x3 minus center)
      - True 3D Game of Life
      - Slice visualization needed
    - **Graph-based topology** - Arbitrary connectivity
      - Cells as nodes, edges define neighbors
      - Support for irregular structures
      - Random/small-world/scale-free networks
  - Implementation challenges:
    - Abstract grid representation to support different topologies
    - Efficient neighbor lookup for each topology type
    - Coordinate system conversions
    - Visualization strategies for non-square grids
    - Pattern file format extensions
  - Rendering approaches:
    - Hexagonal: Offset rows or true hexagon drawing
    - Triangular: Alternating orientations
    - 3D: Orthographic projection, slicing, or VR
    - Graph: Force-directed layout
  - Rule adaptations:
    - Scale B/S rules to different neighbor counts
    - Topology-specific rule variants
    - Investigate emergent behaviors unique to each topology
  - Performance considerations:
    - Maintain vectorized operations where possible
    - Topology-specific optimization strategies
    - GPU acceleration for 3D and large graph topologies

## Performance Improvements

- [ ] Use 3D tensor to parallelize game state computation instead of threads
  - Current implementation uses multiprocessing/threading for parallel pattern search
  - Could leverage PyTorch's tensor operations to run multiple game instances in parallel
  - Stack multiple game grids into a 3D tensor and apply Conway's rules across all at once
  - Would eliminate thread synchronization overhead and better utilize GPU if available

## Detailed Implementation Plan for 3D Tensor Parallelization

### Current Architecture Analysis
- **Grid class** (`core/grid.py`): Uses 2D NumPy arrays with PyTorch for neighbor counting via convolution
- **GameOfLife class** (`core/game.py`): Operates on single Grid instance, applies Conway's rules
- **Parallel search** (`cli.py`): Uses multiprocessing with work queues, each worker creates separate Grid/Game instances
- **Performance bottleneck**: Inter-process communication, redundant Grid creation, CPU-bound processing
- **Complexity issues**: ~700 lines of multiprocessing coordination code, thread management, work distribution

### Proposed 3D Tensor Architecture

#### 1. Create BatchGrid Class
- **Location**: `src/cellular/core/batch_grid.py`
- **Design**:
  ```python
  class BatchGrid:
      def __init__(self, batch_size: int, width: int, height: int, wrap_edges: bool = True, device: str = 'cpu'):
          # Shape: (batch_size, width, height)
          self._cells = torch.zeros(batch_size, width, height, dtype=torch.uint8, device=device)
          self._previous_cells = torch.zeros_like(self._cells)
          # Neighbor counting kernel (same for all batches)
          self._kernel = torch.tensor([[1,1,1],[1,0,1],[1,1,1]], dtype=torch.float32, device=device)
  ```
- **Key methods**:
  - `randomize_batch()`: Initialize all grids with different random states
  - `count_all_neighbors_batch()`: 3D convolution across all grids simultaneously
  - `get_populations()`: Return tensor of population counts for all grids
  - `get_batch_states()`: Return hashed states for cycle detection

#### 2. Create BatchGameOfLife Class
- **Location**: `src/cellular/core/batch_game.py`
- **Design**:
  ```python
  class BatchGameOfLife:
      def __init__(self, batch_grid: BatchGrid):
          self.batch_grid = batch_grid
          self._generations = torch.zeros(batch_grid.batch_size, dtype=torch.int32)
          self._active_mask = torch.ones(batch_grid.batch_size, dtype=torch.bool)
          # State tracking for each game
          self._cycle_detected = torch.zeros(batch_grid.batch_size, dtype=torch.bool)
          self._extinction = torch.zeros(batch_grid.batch_size, dtype=torch.bool)
  ```
- **Key methods**:
  - `step_batch()`: Apply Conway's rules to all active games in parallel
  - `check_termination_batch()`: Check for cycles/extinction across all games
  - `run_until_stable_batch()`: Run all games until termination conditions

#### 3. Refactor CLI Search Implementation
- **Replace**: `parallel_search_for_condition()` with tensor-based implementation
- **Remove**: 
  - `_search_worker_batched()` function
  - All multiprocessing imports and Manager usage
  - Work queue management
  - Progress monitoring threads
- **Key changes**:
  1. Single-process execution with tensor operations
  2. Create one BatchGrid with batch_size = number of parallel searches
  3. Process all searches simultaneously on GPU if available
  4. Use masking to handle terminated games while others continue
  5. Direct pattern harvesting without inter-process communication

#### 4. Implementation Steps
1. **Phase 1**: Implement BatchGrid with basic tensor operations
   - [ ] Create batch_grid.py with 3D tensor storage
   - [ ] Implement batch randomization
   - [ ] Implement 3D convolution for neighbor counting
   - [ ] Add unit tests for BatchGrid

2. **Phase 2**: Implement BatchGameOfLife
   - [ ] Create batch_game.py with vectorized game logic
   - [ ] Implement batch step function with masking
   - [ ] Add batch cycle detection using tensor operations
   - [ ] Add unit tests for BatchGameOfLife

3. **Phase 3**: Replace CLI implementation
   - [ ] Remove multiprocessing code from cli.py
   - [ ] Update parallel_search_for_condition to use BatchGameOfLife
   - [ ] Simplify progress reporting without threads
   - [ ] Remove --workers parameter (no longer needed)
   - [ ] Add --device parameter for CPU/GPU selection
   - [ ] Add performance benchmarking

4. **Phase 4**: Optimization
   - [ ] Add GPU support with device selection
   - [ ] Optimize memory usage with in-place operations
   - [ ] Implement dynamic batch sizing based on available memory
   - [ ] Add mixed precision support for larger batches

### Expected Benefits
1. **Performance**: 10-100x speedup for pattern search (depends on GPU)
2. **Memory efficiency**: Single tensor allocation vs multiple process memory
3. **Scalability**: Can search thousands of patterns simultaneously on GPU
4. **Code simplicity**: Remove ~500+ lines of multiprocessing code
5. **Maintainability**: Single execution path, no race conditions
6. **Debugging**: Easier to debug without inter-process issues

### Migration Strategy
- Replace multiprocessing implementation entirely with tensor-based approach
- Update existing Grid class to support both single and batch operations
- Modify GameOfLife to work with tensor-based grids
- Remove ProcessPoolExecutor and worker-based parallelization
- Simplify CLI by removing worker management code
- All parallel search will use tensor operations (CPU or GPU)

### Testing Strategy
1. Unit tests for BatchGrid tensor operations
2. Unit tests for BatchGameOfLife logic
3. Integration tests comparing results with existing implementation
4. Performance benchmarks on various hardware configurations
5. Memory usage profiling

### Future Extensions
- Support for different rule sets in parallel
- Batch pattern recognition and classification
- Real-time visualization of batch simulations
- Distributed tensor processing across multiple GPUs