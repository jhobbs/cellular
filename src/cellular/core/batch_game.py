"""Batch Game of Life implementation using 3D tensors for parallel simulation."""

from typing import Tuple, Dict, List, Optional
import torch
from collections import deque

from .batch_grid import BatchGrid


class BatchGameOfLife:
    """Conway's Game of Life simulation engine for multiple games in parallel.
    
    This class runs multiple independent Game of Life simulations simultaneously
    using vectorized tensor operations, enabling massive parallelization on GPU.
    """
    
    def __init__(self, batch_grid: BatchGrid, max_history: int = 1000) -> None:
        """Initialize batch game with a batch grid.
        
        Args:
            batch_grid: The BatchGrid containing all game states
            max_history: Maximum generations to track for cycle detection
        """
        self.batch_grid = batch_grid
        self.batch_size = batch_grid.batch_size
        self.device = batch_grid.device
        self.max_history = max_history
        
        # Generation counters for each game
        self._generations = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        
        # Active mask - which games are still running
        self._active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Termination tracking
        self._cycle_detected = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._cycle_length = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self._cycle_start_generation = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self._extinct = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._max_gen_reached = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Population tracking
        self._population_history = deque(maxlen=100)
        self._update_population_history()
        
        # State history for cycle detection - store states for each game
        self._state_history = {}  # game_idx -> deque of (generation, state_bytes)
        for i in range(self.batch_size):
            self._state_history[i] = deque(maxlen=max_history)
            
    @property
    def generations(self) -> torch.Tensor:
        """Current generation numbers for all games."""
        return self._generations
    
    @property
    def populations(self) -> torch.Tensor:
        """Current population counts for all games."""
        return self.batch_grid.populations
    
    @property
    def active_mask(self) -> torch.Tensor:
        """Boolean mask of which games are still active."""
        return self._active_mask
    
    @property
    def active_count(self) -> int:
        """Number of games still running."""
        return int(self._active_mask.sum())
    
    def step_batch(self) -> None:
        """Advance all active simulations by one generation."""
        if self.active_count == 0:
            return
            
        # Save current state before updating
        self.batch_grid.save_state()
        
        # Check for cycles before updating (only for active games)
        if self.active_count > 0:
            self._check_for_cycles_batch()
        
        # Apply Conway's rules to active games
        self._apply_rules_batch()
        
        # Update generation counters for active games
        self._generations[self._active_mask] += 1
        
        # Update population history
        self._update_population_history()
        
        # Check for extinction
        self._check_extinction_batch()
        
    def _apply_rules_batch(self) -> None:
        """Apply Conway's Game of Life rules to all active grids."""
        # Count neighbors for all cells in all grids
        neighbor_counts = self.batch_grid.count_all_neighbors()
        
        # Get current cells
        cells = self.batch_grid.cells
        
        # Apply rules using vectorized operations
        # Birth: dead cell with exactly 3 neighbors
        birth_mask = (cells == 0) & (neighbor_counts == 3)
        
        # Death: live cell with < 2 or > 3 neighbors  
        death_mask = (cells > 0) & ((neighbor_counts < 2) | (neighbor_counts > 3))
        
        # Create new state tensor
        new_cells = cells.clone()
        new_cells[birth_mask] = 1
        new_cells[death_mask] = 0
        
        # Only update active games
        inactive_mask = ~self._active_mask
        if inactive_mask.any():
            # Restore original state for inactive games
            new_cells[inactive_mask] = cells[inactive_mask]
            
        # Update grid
        self.batch_grid._cells = new_cells
        
    def _update_population_history(self) -> None:
        """Update population history for all games."""
        self._population_history.append(self.populations.clone())
        
    def _check_for_cycles_batch(self) -> None:
        """Check all active games for cycles."""
        if self.active_count == 0:
            return
            
        # Get current state bytes
        current_states = self.batch_grid.get_batch_states_bytes()
        
        # Check each active game for cycles
        active_indices = torch.nonzero(self._active_mask).squeeze(-1)
        
        for idx in active_indices:
            idx_val = int(idx)
            current_state = current_states[idx_val]
            current_gen = int(self._generations[idx])
            
            # Check if we've seen this state before
            for past_gen, past_state in self._state_history[idx_val]:
                if past_state == current_state:
                    # Cycle detected!
                    self._cycle_detected[idx] = True
                    self._cycle_length[idx] = current_gen - past_gen
                    self._cycle_start_generation[idx] = past_gen
                    self._active_mask[idx] = False
                    break
                    
            # Add current state to history if no cycle detected
            if self._active_mask[idx]:
                self._state_history[idx_val].append((current_gen, current_state))
                
    def _check_extinction_batch(self) -> None:
        """Check for extinct games (population = 0)."""
        populations = self.populations
        extinct_mask = (populations == 0) & self._active_mask
        
        if extinct_mask.any():
            self._extinct[extinct_mask] = True
            self._active_mask[extinct_mask] = False
            
    def run_until_stable_batch(self, max_generations: int = 10000) -> None:
        """Run all simulations until they become stable, cycle, or go extinct.
        
        Args:
            max_generations: Maximum generations to run each simulation
        """
        while self.active_count > 0:
            # Check if any active games have reached max generations
            at_max_gen = (self._generations >= max_generations) & self._active_mask
            if at_max_gen.any():
                self._max_gen_reached[at_max_gen] = True
                self._active_mask[at_max_gen] = False
                
            if self.active_count == 0:
                break
                
            self.step_batch()
            
    def get_termination_reasons(self) -> List[str]:
        """Get termination reason for each game.
        
        Returns:
            List of strings: 'cycle', 'extinction', 'max_generations', or 'active'
        """
        reasons = []
        for i in range(self.batch_size):
            if self._cycle_detected[i]:
                reasons.append('cycle')
            elif self._extinct[i]:
                reasons.append('extinction')
            elif self._max_gen_reached[i]:
                reasons.append('max_generations')
            elif self._active_mask[i]:
                reasons.append('active')
            else:
                reasons.append('unknown')
        return reasons
    
    def get_statistics_batch(self) -> List[Dict]:
        """Get comprehensive statistics for all simulations.
        
        Returns:
            List of dictionaries with statistics for each game
        """
        stats_list = []
        bboxes = self.batch_grid.get_bounding_boxes()
        
        for i in range(self.batch_size):
            stats = {
                'generation': int(self._generations[i]),
                'population': int(self.populations[i]),
                'cycle_detected': bool(self._cycle_detected[i]),
                'cycle_length': int(self._cycle_length[i]),
                'cycle_start_generation': int(self._cycle_start_generation[i]),
                'grid_size': (self.batch_grid.width, self.batch_grid.height),
                'population_density': float(self.populations[i]) / (self.batch_grid.width * self.batch_grid.height),
            }
            
            # Add bounding box info
            bbox = bboxes[i]
            if bbox[0] != -1:  # Has living cells
                stats['bounding_box'] = (int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2]))  # Convert to (x,y) format
                box_width = int(bbox[3] - bbox[1] + 1)
                box_height = int(bbox[2] - bbox[0] + 1)
                stats['bounding_box_size'] = (box_width, box_height)
                stats['bounding_box_area'] = box_width * box_height
            else:
                stats['bounding_box'] = None
                stats['bounding_box_size'] = (0, 0)
                stats['bounding_box_area'] = 0
                
            stats_list.append(stats)
            
        return stats_list
    
    def reset_batch(self, clear_grids: bool = True) -> None:
        """Reset all simulations.
        
        Args:
            clear_grids: Whether to clear the grids as well
        """
        if clear_grids:
            self.batch_grid.clear()
            
        self._generations.zero_()
        self._active_mask.fill_(True)
        self._cycle_detected.zero_()
        self._cycle_length.zero_()
        self._cycle_start_generation.zero_()
        self._extinct.zero_()
        self._max_gen_reached.zero_()
        
        self._population_history.clear()
        for i in range(self.batch_size):
            self._state_history[i].clear()
            
        self._update_population_history()