#!/usr/bin/env python3
"""
Examples of using the cellular CLI for different scenarios.
"""

import subprocess
import sys


def run_cli_command(args):
    """Run a CLI command and capture its output."""
    cmd = ["cellular-cli"] + args
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        print("-" * 50)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False


def main():
    """Run various CLI examples."""
    print("Conway's Game of Life CLI Examples")
    print("=" * 50)
    
    examples = [
        # Basic examples
        (["--list-patterns"], "List all available patterns"),
        
        # Pattern examples
        (["--pattern", "Block", "-W", "15", "-H", "15", "--show-grid", "--verbose"], 
         "Still life pattern (should be stable)"),
        
        (["--pattern", "Blinker", "-W", "10", "-H", "10", "--show-grid"], 
         "Oscillating blinker pattern"),
        
        (["--pattern", "Glider", "-W", "20", "-H", "20", "--toroidal", "--verbose"], 
         "Glider on toroidal grid"),
        
        # Methuselah examples
        (["--pattern", "Diehard", "-W", "30", "-H", "20"], 
         "Diehard - should die after exactly 130 generations"),
        
        # Random population examples
        (["--width", "40", "--height", "40", "--population", "0.1", "--max-generations", "100"], 
         "Random sparse population"),
        
        (["--width", "20", "--height", "20", "--population", "0.4", "--toroidal", "--max-generations", "50"], 
         "Random dense population on toroidal grid"),
        
        # Speed test
        (["--pattern", "R-pentomino", "-W", "50", "-H", "50", "--verbose"], 
         "R-pentomino performance test"),
    ]
    
    success_count = 0
    for args, description in examples:
        print(f"\nExample: {description}")
        print("-" * len(f"Example: {description}"))
        if run_cli_command(args):
            success_count += 1
        else:
            print("❌ Failed")
            
    print(f"\nSummary: {success_count}/{len(examples)} examples completed successfully")


if __name__ == "__main__":
    main()