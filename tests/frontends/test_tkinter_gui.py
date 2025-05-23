"""Tests for the Tkinter GUI frontend."""

import pytest
import tkinter as tk
from unittest.mock import Mock, patch

from cellular.frontends.tkinter_gui import TkinterGameOfLifeGUI


class TestTkinterGameOfLifeGUI:
    """Test cases for the Tkinter GUI."""

    @pytest.fixture
    def root(self):
        """Create a root Tkinter window for testing."""
        root = tk.Tk()
        root.withdraw()  # Hide the window during tests
        yield root
        root.destroy()

    @pytest.fixture
    def gui(self, root):
        """Create a GUI instance for testing."""
        return TkinterGameOfLifeGUI(root)

    def test_initialization(self, gui):
        """Test GUI initialization."""
        assert gui.cols == 200  # 800 / 4
        assert gui.rows == 150  # 600 / 4
        assert gui.cell_size == 4
        assert gui.frame_rate == 30
        assert gui.running is False
        assert gui.speed_mode is False
        assert gui.grid is not None
        assert gui.game is not None
        assert gui.pattern_library is not None

    def test_toggle_running(self, gui):
        """Test toggling the running state."""
        initial_state = gui.running
        gui.toggle_running()
        assert gui.running != initial_state

        gui.toggle_running()
        assert gui.running == initial_state

    def test_toggle_speed_mode(self, gui):
        """Test toggling speed mode."""
        assert gui.speed_mode is False
        assert gui.speed_btn["text"] == "Speed Mode"
        assert gui.speed_btn["bg"] == "#800080"

        gui.toggle_speed_mode()
        assert gui.speed_mode is True
        assert gui.speed_btn["text"] == "Stop Speed"
        assert gui.speed_btn["bg"] == "#FF4500"
        assert gui.running is True  # Should auto-start

        gui.toggle_speed_mode()
        assert gui.speed_mode is False
        assert gui.speed_btn["text"] == "Speed Mode"
        assert gui.speed_btn["bg"] == "#800080"

    def test_set_frame_rate(self, gui):
        """Test setting frame rate."""
        gui.set_frame_rate("45")
        assert gui.frame_rate == 45
        assert gui.update_interval == 1000 // 45

        gui.set_frame_rate("10.5")
        assert gui.frame_rate == 10

    def test_reset_grid(self, gui):
        """Test grid reset functionality."""
        # Set some cells manually
        gui.grid.set_cell(10, 10, True)
        gui.game.step()  # Advance generation

        initial_generation = gui.game.generation
        assert initial_generation > 0

        # Reset grid
        gui.reset_grid()

        assert gui.game.generation == 0
        assert gui.initial_state is not None
        assert gui.speed_mode is False

    def test_toggle_cell_at_position(self, gui):
        """Test cell toggling via mouse interaction."""
        # Clear grid first to ensure known state
        gui.grid.clear()

        # Calculate canvas position for grid cell (5, 5)
        canvas_x = 5 * gui.cell_size + gui.cell_size // 2
        canvas_y = 5 * gui.cell_size + gui.cell_size // 2

        # Initially cell should be dead
        assert not gui.grid.get_cell(5, 5)

        # Toggle cell
        gui.toggle_cell_at_position(canvas_x, canvas_y)
        assert gui.grid.get_cell(5, 5)

        # Toggle again
        gui.toggle_cell_at_position(canvas_x, canvas_y)
        assert not gui.grid.get_cell(5, 5)

    def test_toggle_cell_out_of_bounds(self, gui):
        """Test cell toggling outside grid bounds."""
        # Position outside canvas
        out_of_bounds_x = gui.canvas_width + 10
        out_of_bounds_y = gui.canvas_height + 10

        # Should not raise error
        gui.toggle_cell_at_position(out_of_bounds_x, out_of_bounds_y)

    def test_on_category_selected(self, gui):
        """Test pattern category selection."""
        # Set category
        gui.pattern_category_var.set("Still Life")
        gui.on_category_selected(None)

        # Check that pattern combo is populated
        values = gui.pattern_combo["values"]
        assert len(values) > 0
        assert "Block" in values

        # Check first pattern is selected
        assert gui.pattern_var.get() in values

    def test_load_selected_pattern(self, gui):
        """Test loading selected pattern."""
        # Select a pattern
        gui.pattern_category_var.set("Oscillators")
        gui.on_category_selected(None)
        gui.pattern_var.set("Blinker")

        # Load pattern
        # initial_generation = gui.game.generation
        gui.load_selected_pattern()

        # Check pattern was loaded
        assert gui.game.generation == 0  # Should reset
        assert gui.running is False  # Should stop running
        assert gui.initial_state is not None
        assert gui.grid.population > 0

    def test_load_selected_pattern_empty(self, gui):
        """Test loading with no pattern selected."""
        gui.pattern_var.set("")

        # Should not raise error
        gui.load_selected_pattern()

    def test_load_selected_pattern_nonexistent(self, gui):
        """Test loading non-existent pattern."""
        gui.pattern_var.set("NonExistentPattern")

        # Should not raise error
        gui.load_selected_pattern()

    def test_draw_cell(self, gui):
        """Test drawing individual cells."""
        # Set cell alive and draw
        gui.grid.set_cell(10, 10, True)
        gui.draw_cell(10, 10)

        # Check that canvas object was created
        assert (10, 10) in gui.cell_objects

        # Set cell dead and draw
        gui.grid.set_cell(10, 10, False)
        gui.draw_cell(10, 10)

        # Check that canvas object was removed
        assert (10, 10) not in gui.cell_objects

    def test_redraw_all_cells(self, gui):
        """Test redrawing all cells."""
        # Set some cells
        gui.grid.set_cell(5, 5, True)
        gui.grid.set_cell(10, 10, True)
        gui.grid.set_cell(15, 15, True)

        # Clear canvas objects
        gui.cell_objects.clear()

        # Redraw all
        gui.redraw_all_cells()

        # Check objects were created for living cells
        assert (5, 5) in gui.cell_objects
        assert (10, 10) in gui.cell_objects
        assert (15, 15) in gui.cell_objects

    def test_update_statistics(self, gui):
        """Test statistics update."""
        # Set up some state
        gui.grid.set_cell(5, 5, True)
        gui.game.step()

        # Update statistics
        gui.update_statistics()

        # Check that labels were updated (basic check)
        assert "Generation:" in gui.stats_labels["Generation"]["text"]
        assert "Population:" in gui.stats_labels["Population"]["text"]
        assert "FPS:" in gui.stats_labels["FPS"]["text"]

    @patch("tkinter.filedialog.asksaveasfilename")
    @patch("builtins.open", create=True)
    @patch("tkinter.messagebox.showinfo")
    def test_save_pattern(self, mock_showinfo, mock_open, mock_filedialog, gui):
        """Test pattern saving."""
        # Set up initial state
        gui.initial_state = {"test": "data"}

        # Mock file dialog
        mock_filedialog.return_value = "/tmp/test.json"

        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Save pattern
        gui.save_pattern()

        # Check that file dialog was called
        mock_filedialog.assert_called_once()

        # Check that file was written
        mock_open.assert_called_once()
        mock_showinfo.assert_called_once()

    @patch("tkinter.filedialog.asksaveasfilename")
    @patch("tkinter.messagebox.showwarning")
    def test_save_pattern_no_initial_state(self, mock_showwarning, mock_filedialog, gui):
        """Test saving pattern with no initial state."""
        gui.initial_state = None

        gui.save_pattern()

        # Should show warning and not open file dialog
        mock_showwarning.assert_called_once()
        mock_filedialog.assert_not_called()

    @patch("tkinter.filedialog.askopenfilename")
    @patch("builtins.open", create=True)
    @patch("tkinter.messagebox.showinfo")
    def test_load_pattern(self, mock_showinfo, mock_open, mock_filedialog, gui):
        """Test pattern loading."""
        # Mock file dialog
        mock_filedialog.return_value = "/tmp/test.json"

        # Mock file content
        test_state = {
            "generation": 0,
            "grid_data": [[0] * gui.rows for _ in range(gui.cols)],
            "grid_width": gui.cols,
            "grid_height": gui.rows,
            "wrap_edges": True,
            "population_history": [0],
            "cycle_detected": False,
            "cycle_length": 0,
            "cycle_start_generation": 0,
        }

        mock_file = Mock()
        mock_file.read.return_value = ""
        mock_open.return_value.__enter__.return_value = mock_file

        with patch("json.load", return_value=test_state):
            gui.load_pattern()

        # Check that file dialog was called
        mock_filedialog.assert_called_once()

        # Check that running was stopped
        assert gui.running is False
        assert gui.initial_state == test_state
        mock_showinfo.assert_called_once()

    @patch("tkinter.filedialog.askopenfilename")
    def test_load_pattern_cancel(self, mock_filedialog, gui):
        """Test canceling pattern load."""
        # Mock canceled file dialog
        mock_filedialog.return_value = ""

        gui.load_pattern()

        # Should not attempt to load anything
        mock_filedialog.assert_called_once()

    def test_mouse_event_simulation(self, gui):
        """Test simulated mouse events."""
        # Clear grid first to ensure known state
        gui.grid.clear()

        # Create mock event
        class MockEvent:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # Test click
        event = MockEvent(50, 50)  # Cell (12, 12) approximately
        gui.on_click(event)

        # Check that corresponding grid cell was affected
        grid_x = event.x // gui.cell_size
        grid_y = event.y // gui.cell_size
        assert gui.grid.get_cell(grid_x, grid_y)

    def test_initial_state_tracking(self, gui):
        """Test that initial state is properly tracked."""
        # At generation 0, changes should update initial state
        assert gui.game.generation == 0

        gui.grid.set_cell(5, 5, True)
        gui.toggle_cell_at_position(5 * gui.cell_size, 5 * gui.cell_size)

        # Initial state should be updated since we're at generation 0
        # (This happens in toggle_cell_at_position)

        # After advancing generation, changes shouldn't update initial state
        gui.game.step()
        old_initial_state = gui.initial_state

        gui.toggle_cell_at_position(10 * gui.cell_size, 10 * gui.cell_size)

        # Initial state should not change after generation > 0
        assert gui.initial_state == old_initial_state
