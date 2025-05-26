# TODO

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