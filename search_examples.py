#!/usr/bin/env python3
"""
Examples of using the CLI search mode to find interesting configurations.
"""

import subprocess
import sys


def run_search_example(title, args):
    """Run a search example and display results."""
    print(f"\n{title}")
    print("=" * len(title))
    cmd = ["cellular-cli"] + args
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Search timed out")
        return False


def main():
    """Run search examples."""
    print("Conway's Game of Life CLI Search Mode Examples")
    print("=" * 50)
    
    examples = [
        (
            "Find patterns with 2-cycle oscillations",
            ["--search", "cycle_length:2", "-W", "20", "-H", "20", "-p", "0.1", "--search-attempts", "50"]
        ),
        (
            "Find long-running patterns (500+ generations)",
            ["--search", "runs_for_at_least:500", "-W", "30", "-H", "30", "-p", "0.12", "--search-attempts", "100"]
        ),
        (
            "Find patterns that go extinct after 30+ generations",
            ["--search", "extinction_after:30", "-W", "15", "-H", "15", "-p", "0.15", "--search-attempts", "30"]
        ),
        (
            "Find patterns that reach high population (80+ cells)",
            ["--search", "population_threshold:80", "-W", "25", "-H", "25", "-p", "0.18", "--search-attempts", "20"]
        ),
        (
            "Find patterns that stabilize with exactly 20 cells",
            ["--search", "stabilizes_with_population:20", "-W", "20", "-H", "20", "-p", "0.12", "--search-attempts", "40"]
        ),
        (
            "Find patterns with large bounding boxes (12x8+)",
            ["--search", "bounding_box_size:12x8", "-W", "25", "-H", "20", "-p", "0.1", "--search-attempts", "30"]
        ),
    ]
    
    success_count = 0
    for title, args in examples:
        if run_search_example(title, args):
            success_count += 1
            
    print(f"\nSummary: {success_count}/{len(examples)} searches completed successfully")
    print("\nSearch conditions available:")
    print("- cycle_length:N          - finishes with N-length cycle")
    print("- runs_for_at_least:N     - runs N+ generations before cycling/stabilizing")
    print("- extinction_after:N      - goes extinct after N+ generations") 
    print("- population_threshold:N  - reaches population of N+ cells at some point")
    print("- stabilizes_with_population:N - stabilizes with exactly N cells")
    print("- bounding_box_size:WxH   - final pattern has bounding box of at least WxH")


if __name__ == "__main__":
    main()