"""Batch grid data structure for parallel cellular automata using 3D tensors."""

from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F


class BatchGrid:
    """Represents multiple 2D grids for cellular automata using 3D tensors.
    
    This class enables parallel simulation of multiple Game of Life instances
    by storing all grids in a single 3D tensor and applying operations across
    all grids simultaneously.
    """
    
    def __init__(
        self, 
        batch_size: int, 
        width: int, 
        height: int, 
        wrap_edges: bool = True,
        device: str = 'cpu'
    ) -> None:
        """Initialize a batch of grids.
        
        Args:
            batch_size: Number of parallel grids
            width: Number of columns in each grid
            height: Number of rows in each grid
            wrap_edges: Whether edges wrap around (toroidal topology)
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.wrap_edges = wrap_edges
        self.device = torch.device(device)
        
        # Initialize 3D tensor for cells: (batch_size, height, width)
        # Note: PyTorch convolution expects (batch, channel, height, width)
        # We'll use height, width order for consistency with convolution
        self._cells = torch.zeros(
            batch_size, height, width, 
            dtype=torch.uint8, 
            device=self.device
        )
        self._previous_cells = torch.zeros_like(self._cells)
        
        # Neighbor counting kernel (same for all batches)
        self._kernel = torch.tensor(
            [[1, 1, 1], 
             [1, 0, 1], 
             [1, 1, 1]], 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
        
    @property
    def cells(self) -> torch.Tensor:
        """Get the current cell tensor."""
        return self._cells
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get grid dimensions as (batch_size, height, width)."""
        return (self.batch_size, self.height, self.width)
    
    def clear(self) -> None:
        """Clear all cells in all grids."""
        self._cells.zero_()
        
    def randomize(self, probability: float = 0.1, seeds: Optional[torch.Tensor] = None) -> None:
        """Randomly populate all grids.
        
        Args:
            probability: Chance each cell will be alive (0.0 to 1.0)
            seeds: Optional tensor of random seeds for each grid in batch
        """
        if seeds is not None:
            # Use different seeds for each grid
            random_values = torch.zeros_like(self._cells, dtype=torch.float32)
            for i in range(self.batch_size):
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(seeds[i]))
                random_values[i] = torch.rand(
                    self.height, self.width, 
                    generator=gen, 
                    device=self.device
                )
        else:
            # Use same random state for all (but different values)
            random_values = torch.rand(
                self.batch_size, self.height, self.width, 
                device=self.device
            )
        
        self._cells = (random_values < probability).to(torch.uint8)
        
    def save_state(self) -> None:
        """Save current state to previous state for all grids."""
        self._previous_cells.copy_(self._cells)
        
    def get_changed_cells(self) -> torch.Tensor:
        """Get mask of cells that changed since last save_state().
        
        Returns:
            Boolean tensor of shape (batch_size, height, width)
        """
        return self._cells != self._previous_cells
    
    @property
    def populations(self) -> torch.Tensor:
        """Get the number of living cells in each grid.
        
        Returns:
            Tensor of shape (batch_size,) with population counts
        """
        return self._cells.sum(dim=(1, 2))
    
    def count_all_neighbors(self) -> torch.Tensor:
        """Count neighbors for all cells in all grids using 3D convolution.
        
        Returns:
            3D tensor with neighbor counts for each cell in each grid
        """
        # Convert to float for convolution
        cells_float = self._cells.float()
        
        # Add channel dimension: (batch, height, width) -> (batch, 1, height, width)
        cells_float = cells_float.unsqueeze(1)
        
        if self.wrap_edges:
            # For toroidal topology, use circular padding
            padded = F.pad(cells_float, (1, 1, 1, 1), mode='circular')
            neighbors = F.conv2d(padded, self._kernel)
        else:
            # For bounded topology, use zero padding
            neighbors = F.conv2d(cells_float, self._kernel, padding=1)
        
        # Remove channel dimension and convert back to int
        # Shape: (batch, 1, height, width) -> (batch, height, width)
        return neighbors.squeeze(1).to(torch.uint8)
    
    def get_batch_states_bytes(self) -> List[bytes]:
        """Get byte representation of each grid state for cycle detection.
        
        Returns:
            List of bytes objects, one per grid
        """
        # Convert each grid to bytes for exact state comparison
        states_bytes = []
        for i in range(self.batch_size):
            # Get the grid as a contiguous byte array
            grid_bytes = self._cells[i].cpu().numpy().tobytes()
            states_bytes.append(grid_bytes)
        return states_bytes
    
    def copy_from_single(self, grid_idx: int, source_cells: torch.Tensor) -> None:
        """Copy a 2D grid into one of the batch grids.
        
        Args:
            grid_idx: Index of grid in batch to copy to
            source_cells: 2D tensor of shape (height, width) to copy
        """
        self._cells[grid_idx] = source_cells.to(self.device)
        
    def extract_single(self, grid_idx: int) -> torch.Tensor:
        """Extract a single 2D grid from the batch.
        
        Args:
            grid_idx: Index of grid to extract
            
        Returns:
            2D tensor of shape (height, width)
        """
        return self._cells[grid_idx].clone()
    
    def get_bounding_boxes(self) -> torch.Tensor:
        """Get bounding boxes of living cells for all grids.
        
        Returns:
            Tensor of shape (batch_size, 4) with (min_y, min_x, max_y, max_x)
            or (-1, -1, -1, -1) for empty grids
        """
        bboxes = torch.full((self.batch_size, 4), -1, dtype=torch.long, device=self.device)
        
        for i in range(self.batch_size):
            living_coords = torch.nonzero(self._cells[i])
            if living_coords.numel() > 0:
                min_coords = living_coords.min(dim=0)[0]
                max_coords = living_coords.max(dim=0)[0]
                bboxes[i] = torch.tensor([
                    min_coords[0], min_coords[1], 
                    max_coords[0], max_coords[1]
                ])
        
        return bboxes