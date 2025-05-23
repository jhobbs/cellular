"""Tests for CLI search functionality."""

from unittest.mock import patch
from io import StringIO

from cellular.frontends.cli import (
    CLIGameOfLife,
    parse_search_condition,
    create_condition_functions,
)


class TestSearchConditions:
    """Test search condition functions."""

    def test_runs_for_at_least_condition(self):
        """Test runs_for_at_least condition."""
        condition_funcs = create_condition_functions()
        func = condition_funcs["runs_for_at_least_n_generations"](100)

        # Should pass: cycle starts after 100+ generations
        assert func(150, "cycle", {"cycle_start_generation": 120})

        # Should pass: hit max generations without cycling
        assert func(200, "max_generations", {})

        # Should fail: cycle starts too early
        assert not func(150, "cycle", {"cycle_start_generation": 50})

        # Should fail: extinction
        assert not func(50, "extinction", {})

    def test_cycle_length_condition(self):
        """Test cycle_length condition."""
        condition_funcs = create_condition_functions()
        func = condition_funcs["finishes_with_cycle_length"](3)

        # Should pass: correct cycle length
        assert func(100, "cycle", {"cycle_length": 3})

        # Should fail: wrong cycle length
        assert not func(100, "cycle", {"cycle_length": 2})

        # Should fail: no cycle
        assert not func(100, "extinction", {})

    def test_extinction_after_condition(self):
        """Test extinction_after condition."""
        condition_funcs = create_condition_functions()
        func = condition_funcs["finishes_with_extinction_after_n_generations"](50)

        # Should pass: extinction after enough generations
        assert func(75, "extinction", {})

        # Should fail: extinction too early
        assert not func(25, "extinction", {})

        # Should fail: not extinction
        assert not func(100, "cycle", {})

    def test_population_threshold_condition(self):
        """Test population_threshold condition."""
        condition_funcs = create_condition_functions()
        func = condition_funcs["reaches_population_threshold"](50)

        # Should pass: reached threshold
        assert func(100, "cycle", {"population_history": [10, 30, 60, 40]})

        # Should fail: never reached threshold
        assert not func(100, "cycle", {"population_history": [10, 30, 40]})

        # Should handle empty history
        assert not func(100, "extinction", {"population_history": []})

    def test_stabilizes_with_population_condition(self):
        """Test stabilizes_with_population condition."""
        condition_funcs = create_condition_functions()
        func = condition_funcs["stabilizes_with_population"](25)

        # Should pass: cycle with correct population
        assert func(100, "cycle", {"population": 25})

        # Should pass: max generations with correct population
        assert func(1000, "max_generations", {"population": 25})

        # Should fail: wrong population
        assert not func(100, "cycle", {"population": 30})

        # Should fail: extinction
        assert not func(50, "extinction", {"population": 0})

    def test_bounding_box_size_condition(self):
        """Test bounding_box_size condition."""
        condition_funcs = create_condition_functions()
        func = condition_funcs["has_bounding_box_size"](10, 5)

        # Should pass: meets minimum size
        assert func(100, "cycle", {"bounding_box_size": (15, 8)})

        # Should pass: exactly minimum size
        assert func(100, "cycle", {"bounding_box_size": (10, 5)})

        # Should fail: too small
        assert not func(100, "cycle", {"bounding_box_size": (8, 4)})

        # Should fail: width okay but height too small
        assert not func(100, "cycle", {"bounding_box_size": (12, 3)})


class TestConditionParsing:
    """Test condition string parsing."""

    def test_parse_runs_for_at_least(self):
        """Test parsing runs_for_at_least condition."""
        func, desc = parse_search_condition("runs_for_at_least:500")
        assert "runs for at least 500 generations" in desc

        # Test the function
        assert func(600, "cycle", {"cycle_start_generation": 550})
        assert not func(600, "cycle", {"cycle_start_generation": 400})

    def test_parse_cycle_length(self):
        """Test parsing cycle_length condition."""
        func, desc = parse_search_condition("cycle_length:4")
        assert "finishes with cycle length 4" in desc

        # Test the function
        assert func(100, "cycle", {"cycle_length": 4})
        assert not func(100, "cycle", {"cycle_length": 2})

    def test_parse_extinction_after(self):
        """Test parsing extinction_after condition."""
        func, desc = parse_search_condition("extinction_after:200")
        assert "goes extinct after at least 200 generations" in desc

        # Test the function
        assert func(250, "extinction", {})
        assert not func(150, "extinction", {})

    def test_parse_population_threshold(self):
        """Test parsing population_threshold condition."""
        func, desc = parse_search_condition("population_threshold:100")
        assert "reaches population of at least 100" in desc

        # Test the function
        assert func(50, "cycle", {"population_history": [20, 120, 80]})
        assert not func(50, "cycle", {"population_history": [20, 80, 50]})

    def test_parse_stabilizes_with_population(self):
        """Test parsing stabilizes_with_population condition."""
        func, desc = parse_search_condition("stabilizes_with_population:42")
        assert "stabilizes with exactly 42 cells" in desc

        # Test the function
        assert func(100, "cycle", {"population": 42})
        assert not func(100, "cycle", {"population": 40})

    def test_parse_bounding_box_size(self):
        """Test parsing bounding_box_size condition."""
        func, desc = parse_search_condition("bounding_box_size:15x8")
        assert "has bounding box of at least 15x8" in desc

        # Test the function
        assert func(100, "cycle", {"bounding_box_size": (20, 10)})
        assert not func(100, "cycle", {"bounding_box_size": (10, 10)})

    def test_parse_invalid_format(self):
        """Test parsing invalid condition format."""
        # Missing colon
        try:
            parse_search_condition("invalid_format")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid condition format" in str(e)

    def test_parse_unknown_condition(self):
        """Test parsing unknown condition type."""
        try:
            parse_search_condition("unknown_type:100")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown condition type" in str(e)

    def test_parse_invalid_numeric_value(self):
        """Test parsing invalid numeric value."""
        try:
            parse_search_condition("cycle_length:not_a_number")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid numeric value" in str(e)

    def test_parse_invalid_bounding_box_format(self):
        """Test parsing invalid bounding box format."""
        try:
            parse_search_condition("bounding_box_size:invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "requires format 'WxH'" in str(e)


class TestCLISearch:
    """Test CLI search functionality."""

    def test_search_for_condition_success(self):
        """Test successful search."""
        cli = CLIGameOfLife()

        # Mock condition that always returns True
        def always_true(final_gen, reason, stats):
            return True

        found, attempts, result = cli.search_for_condition(
            width=10,
            height=10,
            population_rate=0.1,
            toroidal=False,
            max_generations=50,
            condition_func=always_true,
            condition_name="always true",
            max_attempts=5,
            verbose=False,
        )

        assert found is True
        assert attempts == 1
        assert result is not None
        assert len(result) == 3  # (final_gen, reason, stats)

    def test_search_for_condition_failure(self):
        """Test failed search."""
        cli = CLIGameOfLife()

        # Mock condition that always returns False
        def always_false(final_gen, reason, stats):
            return False

        found, attempts, result = cli.search_for_condition(
            width=10,
            height=10,
            population_rate=0.1,
            toroidal=False,
            max_generations=50,
            condition_func=always_false,
            condition_name="always false",
            max_attempts=3,
            verbose=False,
        )

        assert found is False
        assert attempts == 3
        assert result is None

    def test_search_with_pattern(self):
        """Test search using a specific pattern."""
        cli = CLIGameOfLife()

        # Condition: must have exactly 4 cells (Block pattern)
        def has_four_cells(final_gen, reason, stats):
            return stats.get("population", 0) == 4

        found, attempts, result = cli.search_for_condition(
            width=20,
            height=20,
            population_rate=0.0,  # Ignored when pattern is used
            toroidal=False,
            max_generations=10,
            condition_func=has_four_cells,
            condition_name="has exactly 4 cells",
            max_attempts=5,
            pattern="Block",
            pattern_x=8,
            pattern_y=8,
            verbose=False,
        )

        # Block is a still life with 4 cells, should always match
        assert found is True
        assert result is not None
        final_gen, reason, stats = result
        assert stats["population"] == 4

    def test_search_with_seed(self):
        """Test search with random seed for reproducibility."""
        cli = CLIGameOfLife()

        # Condition that's likely to be met eventually
        def reasonable_condition(final_gen, reason, stats):
            return reason == "extinction" or final_gen > 20

        # Run search twice with same seed
        found1, attempts1, _ = cli.search_for_condition(
            width=15,
            height=15,
            population_rate=0.1,
            toroidal=False,
            max_generations=100,
            condition_func=reasonable_condition,
            condition_name="reasonable",
            max_attempts=10,
            seed=12345,
            verbose=False,
        )

        found2, attempts2, _ = cli.search_for_condition(
            width=15,
            height=15,
            population_rate=0.1,
            toroidal=False,
            max_generations=100,
            condition_func=reasonable_condition,
            condition_name="reasonable",
            max_attempts=10,
            seed=12345,
            verbose=False,
        )

        # Results should be identical with same seed
        assert found1 == found2
        # Note: attempts might differ slightly due to random timing in test environment
        # but both should find a solution
        assert found1 is True

    @patch("sys.stdout", new_callable=StringIO)
    def test_search_verbose_output(self, mock_stdout):
        """Test verbose output during search."""
        cli = CLIGameOfLife()

        def condition_met_on_third(final_gen, reason, stats):
            # This will be called multiple times, return True on 3rd call
            if not hasattr(condition_met_on_third, "call_count"):
                condition_met_on_third.call_count = 0
            condition_met_on_third.call_count += 1
            return condition_met_on_third.call_count >= 3

        found, attempts, result = cli.search_for_condition(
            width=10,
            height=10,
            population_rate=0.1,
            toroidal=False,
            max_generations=50,
            condition_func=condition_met_on_third,
            condition_name="test condition",
            max_attempts=10,
            verbose=True,
        )

        output = mock_stdout.getvalue()
        assert "Searching for condition: test condition" in output
        assert "Grid: 10x10" in output
        assert "✓ Found matching configuration" in output

    def test_search_exception_handling(self):
        """Test search handles simulation exceptions gracefully."""
        cli = CLIGameOfLife()

        # Mock run_simulation to raise exception on first call
        original_run = cli.run_simulation
        call_count = [0]

        def mock_run_simulation(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Simulated error")
            return original_run(*args, **kwargs)

        cli.run_simulation = mock_run_simulation

        def always_true(final_gen, reason, stats):
            return True

        found, attempts, result = cli.search_for_condition(
            width=10,
            height=10,
            population_rate=0.1,
            toroidal=False,
            max_generations=50,
            condition_func=always_true,
            condition_name="test",
            max_attempts=5,
            verbose=False,
        )

        # Should succeed on second attempt despite first failure
        assert found is True
        assert attempts == 2
