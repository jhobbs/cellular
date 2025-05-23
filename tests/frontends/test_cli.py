"""Tests for the CLI frontend."""

import argparse
from unittest.mock import Mock, patch
from io import StringIO

from cellular.frontends.cli import (
    CLIGameOfLife,
    create_parser,
    format_finish_reason,
    print_results,
    validate_args,
    main,
)


class TestCLIGameOfLife:
    """Test cases for the CLI Game of Life."""

    def test_initialization(self):
        """Test CLI initialization."""
        cli = CLIGameOfLife()
        assert cli.pattern_library is not None
        assert len(cli.pattern_library.list_patterns()) > 0

    def test_run_simulation_random(self):
        """Test running simulation with random population."""
        cli = CLIGameOfLife()

        final_gen, reason, stats = cli.run_simulation(
            width=10,
            height=10,
            population_rate=0.1,
            toroidal=True,
            max_generations=100,
            verbose=False,
            show_grid=False,
        )

        assert isinstance(final_gen, int)
        assert final_gen >= 0
        assert reason in ["extinction", "cycle", "max_generations"]
        assert isinstance(stats, dict)
        assert "generation" in stats
        assert "population" in stats
        assert "duration_seconds" in stats

    def test_run_simulation_with_pattern(self):
        """Test running simulation with a specific pattern."""
        cli = CLIGameOfLife()

        final_gen, reason, stats = cli.run_simulation(
            width=20,
            height=20,
            population_rate=0.0,  # Will be ignored when pattern is used
            toroidal=False,
            max_generations=50,
            pattern="Blinker",
            pattern_x=10,
            pattern_y=10,
            verbose=False,
            show_grid=False,
        )

        assert isinstance(final_gen, int)
        assert reason in ["extinction", "cycle", "max_generations"]
        assert stats["initial_population"] == 3  # Blinker has 3 cells

    def test_run_simulation_invalid_pattern(self):
        """Test running simulation with invalid pattern falls back to random."""
        cli = CLIGameOfLife()

        final_gen, reason, stats = cli.run_simulation(
            width=10,
            height=10,
            population_rate=0.2,
            toroidal=False,
            max_generations=10,
            pattern="NonExistentPattern",
            verbose=False,
            show_grid=False,
        )

        # Should fall back to random population
        assert isinstance(final_gen, int)
        assert stats["initial_population"] > 0  # Should have some random cells

    def test_format_grid_small(self):
        """Test grid formatting for small grids."""
        cli = CLIGameOfLife()
        from cellular.core.grid import Grid

        grid = Grid(5, 5)
        grid.set_cell(2, 2, True)

        formatted = cli._format_grid(grid)
        assert "*" in formatted
        assert "....." in formatted

    def test_format_grid_large(self):
        """Test grid formatting for large grids."""
        cli = CLIGameOfLife()
        from cellular.core.grid import Grid

        grid = Grid(100, 100)
        formatted = cli._format_grid(grid, max_size=50)
        assert "too large to display" in formatted

    @patch("sys.stdout", new_callable=StringIO)
    def test_list_patterns(self, mock_stdout):
        """Test pattern listing."""
        cli = CLIGameOfLife()
        cli.list_patterns()

        output = mock_stdout.getvalue()
        assert "Available patterns:" in output
        assert "Block" in output
        assert "Blinker" in output
        assert "Still Life:" in output
        assert "Oscillators:" in output


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Test default values
        args = parser.parse_args([])
        assert args.width == 50
        assert args.height == 50
        assert args.population == 0.1
        assert args.toroidal is False
        assert args.max_generations == 10000

    def test_parse_basic_args(self):
        """Test parsing basic arguments."""
        parser = create_parser()

        args = parser.parse_args(
            ["--width", "30", "--height", "40", "--population", "0.2", "--toroidal"]
        )

        assert args.width == 30
        assert args.height == 40
        assert args.population == 0.2
        assert args.toroidal is True

    def test_parse_pattern_args(self):
        """Test parsing pattern-related arguments."""
        parser = create_parser()

        args = parser.parse_args(
            ["--pattern", "Glider", "--pattern-x", "10", "--pattern-y", "15"]
        )

        assert args.pattern == "Glider"
        assert args.pattern_x == 10
        assert args.pattern_y == 15

    def test_parse_output_args(self):
        """Test parsing output-related arguments."""
        parser = create_parser()

        args = parser.parse_args(
            ["--verbose", "--show-grid", "--max-generations", "5000"]
        )

        assert args.verbose is True
        assert args.show_grid is True
        assert args.max_generations == 5000

    def test_parse_short_args(self):
        """Test parsing short argument forms."""
        parser = create_parser()

        args = parser.parse_args(
            ["-W", "25", "-H", "35", "-p", "0.15", "-t", "-m", "1000", "-v", "-g"]
        )

        assert args.width == 25
        assert args.height == 35
        assert args.population == 0.15
        assert args.toroidal is True
        assert args.max_generations == 1000
        assert args.verbose is True
        assert args.show_grid is True


class TestValidation:
    """Test argument validation."""

    def test_validate_args_valid(self):
        """Test validation with valid arguments."""
        args = argparse.Namespace(
            width=50,
            height=50,
            population=0.1,
            max_generations=1000,
            pattern_x=0,
            pattern_y=0,
        )

        assert validate_args(args) is True

    def test_validate_args_invalid_width(self):
        """Test validation with invalid width."""
        args = argparse.Namespace(
            width=-5,
            height=50,
            population=0.1,
            max_generations=1000,
            pattern_x=0,
            pattern_y=0,
        )

        assert validate_args(args) is False

    def test_validate_args_invalid_population(self):
        """Test validation with invalid population rate."""
        args = argparse.Namespace(
            width=50,
            height=50,
            population=1.5,
            max_generations=1000,
            pattern_x=0,
            pattern_y=0,
        )

        assert validate_args(args) is False

    def test_validate_args_negative_offset(self):
        """Test validation with negative pattern offset."""
        args = argparse.Namespace(
            width=50,
            height=50,
            population=0.1,
            max_generations=1000,
            pattern_x=-5,
            pattern_y=0,
        )

        assert validate_args(args) is False


class TestFormatting:
    """Test output formatting functions."""

    def test_format_finish_reason_extinction(self):
        """Test formatting extinction reason."""
        reason = format_finish_reason("extinction", {})
        assert "Extinction" in reason
        assert "died" in reason

    def test_format_finish_reason_cycle(self):
        """Test formatting cycle detection reason."""
        stats = {"cycle_length": 3, "cycle_start_generation": 15}
        reason = format_finish_reason("cycle", stats)
        assert "Cycle detected" in reason
        assert "length 3" in reason
        assert "generation 15" in reason

    def test_format_finish_reason_max_generations(self):
        """Test formatting max generations reason."""
        stats = {"generation": 10000}
        reason = format_finish_reason("max_generations", stats)
        assert "Maximum generations" in reason
        assert "10000" in reason

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_results_verbose(self, mock_stdout):
        """Test printing results in verbose mode."""
        stats = {
            "grid_size": (50, 50),
            "initial_population": 250,
            "population": 100,
            "population_density": 0.04,
            "population_change_rate": -1.5,
            "duration_seconds": 2.5,
            "generations_per_second": 400.0,
            "bounding_box": (10, 10, 40, 40),
            "bounding_box_size": (31, 31),
        }

        print_results(1000, "cycle", stats, verbose=True)

        output = mock_stdout.getvalue()
        assert "1000 generations" in output
        assert "Cycle detected" in output
        assert "Grid size: 50x50" in output
        assert "Initial population: 250" in output
        assert "2.500 seconds" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_results_compact(self, mock_stdout):
        """Test printing results in compact mode."""
        stats = {
            "initial_population": 250,
            "population": 100,
            "duration_seconds": 2.5,
            "generations_per_second": 400.0,
        }

        print_results(1000, "extinction", stats, verbose=False)

        output = mock_stdout.getvalue()
        assert "1000 generations" in output
        assert "250 → 100" in output  # Population change
        assert "2.500s" in output
        assert "400 gen/s" in output


class TestMainFunction:
    """Test the main CLI function."""

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_list_patterns(self, mock_cli_class):
        """Test main function with --list-patterns."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli

        with patch("sys.argv", ["cellular-cli", "--list-patterns"]):
            result = main()

        assert result == 0
        mock_cli.list_patterns.assert_called_once()

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_invalid_args(self, mock_cli_class):
        """Test main function with invalid arguments."""
        with patch("sys.argv", ["cellular-cli", "--width", "-5"]):
            result = main()

        assert result == 1

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_invalid_pattern(self, mock_cli_class):
        """Test main function with invalid pattern."""
        mock_cli = Mock()
        mock_cli.pattern_library.get_pattern.return_value = None
        mock_cli.pattern_library.list_patterns.return_value = ["Block", "Blinker"]
        mock_cli_class.return_value = mock_cli

        with patch("sys.argv", ["cellular-cli", "--pattern", "InvalidPattern"]):
            result = main()

        assert result == 1

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_successful_run(self, mock_cli_class):
        """Test successful simulation run."""
        mock_cli = Mock()
        mock_cli.run_simulation.return_value = (
            100,
            "cycle",
            {
                "grid_size": (20, 20),
                "initial_population": 10,
                "population": 8,
                "population_density": 0.02,
                "population_change_rate": 0.0,
                "duration_seconds": 0.5,
                "generations_per_second": 200.0,
                "bounding_box": None,
            },
        )
        mock_cli_class.return_value = mock_cli

        with patch("sys.argv", ["cellular-cli"]):
            result = main()

        assert result == 0
        mock_cli.run_simulation.assert_called_once()

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_pattern_auto_center(self, mock_cli_class):
        """Test automatic pattern centering."""
        mock_cli = Mock()
        mock_pattern = Mock()
        mock_pattern.get_size.return_value = (5, 3)
        mock_cli.pattern_library.get_pattern.return_value = mock_pattern
        mock_cli.run_simulation.return_value = (
            50,
            "extinction",
            {
                "initial_population": 5,
                "population": 0,
                "duration_seconds": 0.1,
                "generations_per_second": 500.0,
            },
        )
        mock_cli_class.return_value = mock_cli

        with patch(
            "sys.argv",
            [
                "cellular-cli",
                "--pattern",
                "TestPattern",
                "--width",
                "30",
                "--height",
                "20",
            ],
        ):
            result = main()

        assert result == 0

        # Check that run_simulation was called with centered coordinates
        call_args = mock_cli.run_simulation.call_args
        assert call_args[1]["pattern_x"] == 12  # (30 - 5) // 2
        assert call_args[1]["pattern_y"] == 8  # (20 - 3) // 2

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_keyboard_interrupt(self, mock_cli_class):
        """Test handling of keyboard interrupt."""
        mock_cli = Mock()
        mock_cli.run_simulation.side_effect = KeyboardInterrupt()
        mock_cli_class.return_value = mock_cli

        with patch("sys.argv", ["cellular-cli"]):
            result = main()

        assert result == 1

    @patch("cellular.frontends.cli.CLIGameOfLife")
    def test_main_exception(self, mock_cli_class):
        """Test handling of general exceptions."""
        mock_cli = Mock()
        mock_cli.run_simulation.side_effect = Exception("Test error")
        mock_cli_class.return_value = mock_cli

        with patch("sys.argv", ["cellular-cli"]):
            result = main()

        assert result == 1
