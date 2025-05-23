#!/usr/bin/env python3
"""
Example usage of the cellular automata package.
"""

from cellular import Grid, GameOfLife, PatternLibrary


def main():
    """Demonstrate programmatic usage of the cellular package."""
    # Create a grid and game
    grid = Grid(20, 20)
    game = GameOfLife(grid)
    
    # Load a pattern
    library = PatternLibrary()
    glider = library.get_pattern("Glider")
    
    if glider:
        # Apply glider pattern to the center of the grid
        glider.apply_to_grid(grid, offset_x=8, offset_y=8)
        
        print("Initial state:")
        print(grid)
        print(f"Population: {game.population}")
        print()
        
        # Run simulation for 10 generations
        for i in range(10):
            game.step()
            print(f"Generation {game.generation}:")
            print(grid)
            print(f"Population: {game.population}")
            
            if game.cycle_detected:
                print(f"Cycle detected! Length: {game.cycle_length}")
                break
            
            print()
    
    # Show statistics
    stats = game.get_statistics()
    print("Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()