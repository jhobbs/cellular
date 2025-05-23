"""Tkinter GUI frontend for Conway's Game of Life."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Tuple, Any
from collections import deque
import json
import os

from ..core.grid import Grid
from ..core.game import GameOfLife
from ..core.patterns import PatternLibrary


class TkinterGameOfLifeGUI:
    """Tkinter-based GUI for Conway's Game of Life."""

    def __init__(self, master: tk.Tk) -> None:
        """Initialize the GUI.

        Args:
            master: Root Tkinter window
        """
        self.master = master
        self.master.title("Conway's Game of Life")
        self.master.configure(bg="#333333")

        # Display parameters
        self.cell_size = 4
        self.canvas_width = 800
        self.canvas_height = 600
        self.cols = self.canvas_width // self.cell_size
        self.rows = self.canvas_height // self.cell_size

        # Initialize core components
        self.grid = Grid(self.cols, self.rows, wrap_edges=True)
        self.game = GameOfLife(self.grid)
        self.pattern_library = PatternLibrary()

        # GUI state
        self.running = False
        self.frame_rate = 30
        self.update_interval = 1000 // self.frame_rate
        self.last_update = 0
        self.initial_population = 0.09

        # Speed mode for rapid simulation
        self.speed_mode = False
        self.max_generations = 100000

        # Canvas objects cache for efficient rendering
        self.cell_objects: Dict[Tuple[int, int], int] = {}

        # Performance tracking
        self.frame_times: deque = deque(maxlen=30)

        # Pattern state tracking
        self.initial_state: Optional[Dict] = None

        self.setup_ui()
        self.reset_grid()
        self.update_loop()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        # Main control frame
        control_frame = tk.Frame(self.master, bg="#333333")
        control_frame.pack(pady=5)

        # Control buttons
        self._create_control_buttons(control_frame)

        # Main content area
        main_frame = tk.Frame(self.master, bg="#333333")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas area
        self._create_canvas(main_frame)

        # Control panel
        self._create_control_panel(main_frame)

    def _create_control_buttons(self, parent: tk.Frame) -> None:
        """Create the main control buttons."""
        self.toggle_btn = tk.Button(
            parent,
            text="Toggle Run",
            command=self.toggle_running,
            bg="#555555",
            fg="white",
            font=("Arial", 9),
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=3)

        self.reset_btn = tk.Button(
            parent,
            text="Reset",
            command=self.reset_grid,
            bg="#555555",
            fg="white",
            font=("Arial", 9),
        )
        self.reset_btn.pack(side=tk.LEFT, padx=3)

        self.save_btn = tk.Button(
            parent,
            text="Save Pattern",
            command=self.save_pattern,
            bg="#444444",
            fg="white",
            font=("Arial", 9),
        )
        self.save_btn.pack(side=tk.LEFT, padx=3)

        self.load_btn = tk.Button(
            parent,
            text="Load Pattern",
            command=self.load_pattern,
            bg="#444444",
            fg="white",
            font=("Arial", 9),
        )
        self.load_btn.pack(side=tk.LEFT, padx=3)

        self.save_current_btn = tk.Button(
            parent,
            text="Save Current",
            command=self.save_current_state,
            bg="#666666",
            fg="white",
            font=("Arial", 8),
        )
        self.save_current_btn.pack(side=tk.LEFT, padx=2)

        self.speed_btn = tk.Button(
            parent,
            text="Speed Mode",
            command=self.toggle_speed_mode,
            bg="#800080",
            fg="white",
            font=("Arial", 9),
        )
        self.speed_btn.pack(side=tk.LEFT, padx=3)

    def _create_canvas(self, parent: tk.Frame) -> None:
        """Create the game canvas."""
        canvas_frame = tk.Frame(parent, bg="#333333")
        canvas_frame.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black",
            highlightthickness=1,
            highlightbackground="white",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def _create_control_panel(self, parent: tk.Frame) -> None:
        """Create the control panel with sliders and stats."""
        controls_frame = tk.Frame(parent, bg="#333333")
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

        # Population slider
        tk.Label(
            controls_frame,
            text="Initial Population:",
            bg="#333333",
            fg="white",
            font=("Arial", 9),
        ).pack(anchor="w")
        self.pop_slider = tk.Scale(
            controls_frame,
            from_=0,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            bg="#555555",
            fg="white",
            font=("Arial", 8),
            length=180,
        )
        self.pop_slider.set(self.initial_population)
        self.pop_slider.pack(pady=(0, 10))

        # Frame rate slider
        tk.Label(
            controls_frame,
            text="Frame Rate:",
            bg="#333333",
            fg="white",
            font=("Arial", 9),
        ).pack(anchor="w")
        self.rate_slider = tk.Scale(
            controls_frame,
            from_=1,
            to=60,
            orient=tk.HORIZONTAL,
            bg="#555555",
            fg="white",
            font=("Arial", 8),
            length=180,
            command=self.set_frame_rate,
        )
        self.rate_slider.set(self.frame_rate)
        self.rate_slider.pack(pady=(0, 20))

        # Pattern selection
        self._create_pattern_selector(controls_frame)

        # Statistics display
        self._create_statistics_display(controls_frame)

    def _create_pattern_selector(self, parent: tk.Frame) -> None:
        """Create pattern selection controls."""
        tk.Label(
            parent,
            text="Patterns:",
            bg="#333333",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        # Pattern category dropdown
        self.pattern_category_var = tk.StringVar()
        self.pattern_category_combo = ttk.Combobox(
            parent,
            textvariable=self.pattern_category_var,
            values=list(self.pattern_library.get_patterns_by_category().keys()),
            state="readonly",
            width=20,
        )
        self.pattern_category_combo.pack(pady=2)
        self.pattern_category_combo.bind(
            "<<ComboboxSelected>>", self.on_category_selected
        )

        # Pattern selection dropdown
        self.pattern_var = tk.StringVar()
        self.pattern_combo = ttk.Combobox(
            parent, textvariable=self.pattern_var, state="readonly", width=20
        )
        self.pattern_combo.pack(pady=2)

        # Load pattern button
        load_pattern_btn = tk.Button(
            parent,
            text="Load Selected Pattern",
            command=self.load_selected_pattern,
            bg="#006400",
            fg="white",
            font=("Arial", 8),
        )
        load_pattern_btn.pack(pady=5)

        # Initialize with first category
        categories = list(self.pattern_library.get_patterns_by_category().keys())
        if categories:
            self.pattern_category_var.set(categories[0])
            self.on_category_selected(None)

    def _create_statistics_display(self, parent: tk.Frame) -> None:
        """Create the statistics display area."""
        tk.Label(
            parent,
            text="Statistics:",
            bg="#333333",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(20, 5))

        self.stats_labels: Dict[str, tk.Label] = {}
        stats = [
            "Running",
            "Population",
            "Rate Change",
            "Generation",
            "Cycle Status",
            "Speed Mode",
        ]

        for stat in stats:
            label = tk.Label(
                parent,
                text=f"{stat}: ",
                bg="#333333",
                fg="white",
                font=("Arial", 9),
                anchor="w",
            )
            label.pack(anchor="w", pady=1)
            self.stats_labels[stat] = label

        # FPS display
        self.stats_labels["FPS"] = tk.Label(
            parent,
            text="FPS: 0",
            bg="#333333",
            fg="#00FF00",
            font=("Arial", 9),
            anchor="w",
        )
        self.stats_labels["FPS"].pack(anchor="w", pady=1)

    def on_category_selected(self, event: Optional[Any]) -> None:
        """Handle pattern category selection."""
        category = self.pattern_category_var.get()
        if not category:
            return

        patterns_by_category = self.pattern_library.get_patterns_by_category()
        patterns = patterns_by_category.get(category, [])

        self.pattern_combo["values"] = patterns
        if patterns:
            self.pattern_var.set(patterns[0])

    def load_selected_pattern(self) -> None:
        """Load the currently selected pattern."""
        pattern_name = self.pattern_var.get()
        if not pattern_name:
            return

        pattern = self.pattern_library.get_pattern(pattern_name)
        if pattern:
            self.running = False

            # Center the pattern on the grid
            pattern_normalized = pattern.normalize()
            offset_x = (self.cols - pattern_normalized.get_size()[0]) // 2
            offset_y = (self.rows - pattern_normalized.get_size()[1]) // 2

            pattern_normalized.apply_to_grid(self.grid, offset_x, offset_y)
            self.game.reset(clear_grid=False)
            self.initial_state = self.game.save_state()

            self.redraw_all_cells()

    def set_frame_rate(self, value: str) -> None:
        """Set the frame rate from slider."""
        self.frame_rate = int(float(value))
        self.update_interval = 1000 // self.frame_rate

    def toggle_running(self) -> None:
        """Toggle the simulation running state."""
        self.running = not self.running

    def toggle_speed_mode(self) -> None:
        """Toggle speed mode for rapid simulation."""
        self.speed_mode = not self.speed_mode

        if self.speed_mode:
            self.speed_btn.config(text="Stop Speed", bg="#FF4500")
            if not self.running:
                self.running = True
        else:
            self.speed_btn.config(text="Speed Mode", bg="#800080")

    def reset_grid(self) -> None:
        """Reset the grid with random population."""
        self.initial_population = self.pop_slider.get()
        self.grid.randomize(self.initial_population)
        self.game.reset(clear_grid=False)

        # Save initial state
        self.initial_state = self.game.save_state()

        # Reset speed mode
        if self.speed_mode:
            self.speed_mode = False
            self.speed_btn.config(text="Speed Mode", bg="#800080")

        self.redraw_all_cells()

    def on_click(self, event: tk.Event) -> None:
        """Handle mouse click on canvas."""
        self.toggle_cell_at_position(event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        """Handle mouse drag on canvas."""
        self.toggle_cell_at_position(event.x, event.y)

    def toggle_cell_at_position(self, canvas_x: int, canvas_y: int) -> None:
        """Toggle cell at canvas coordinates."""
        grid_x = canvas_x // self.cell_size
        grid_y = canvas_y // self.cell_size

        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            self.grid.toggle_cell(grid_x, grid_y)
            self.draw_cell(grid_x, grid_y)

            # Update initial state if at generation 0
            if self.game.generation == 0:
                self.initial_state = self.game.save_state()

    def draw_cell(self, x: int, y: int) -> None:
        """Draw or update a single cell on the canvas."""
        cell_key = (x, y)
        x1 = x * self.cell_size
        y1 = y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size

        if self.grid.get_cell(x, y):
            color = "#00FF00"
            if cell_key in self.cell_objects:
                self.canvas.itemconfig(self.cell_objects[cell_key], fill=color)
            else:
                obj = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=""
                )
                self.cell_objects[cell_key] = obj
        else:
            if cell_key in self.cell_objects:
                self.canvas.delete(self.cell_objects[cell_key])
                del self.cell_objects[cell_key]

    def redraw_all_cells(self) -> None:
        """Redraw all cells on the canvas."""
        self.canvas.delete("all")
        self.cell_objects.clear()

        for x in range(self.cols):
            for y in range(self.rows):
                if self.grid.get_cell(x, y):
                    self.draw_cell(x, y)

    def draw_changed_cells(self) -> None:
        """Draw only the cells that changed since last update."""
        for x, y in self.grid.get_changed_cells():
            self.draw_cell(x, y)

    def update_statistics(self) -> None:
        """Update the statistics display."""
        stats = self.game.get_statistics()

        # Calculate FPS
        current_time = self.master.tk.call("clock", "milliseconds")
        self.frame_times.append(current_time)
        if len(self.frame_times) > 1:
            fps = (
                1000
                * (len(self.frame_times) - 1)
                / (self.frame_times[-1] - self.frame_times[0])
            )
        else:
            fps = 0

        # Format cycle status
        cycle_status = "None"
        cycle_color = "#FFFFFF"
        if stats["cycle_detected"]:
            cycle_status = (
                f"Cycle {stats['cycle_length']} (gen {stats['cycle_start_generation']})"
            )
            cycle_color = "#FFD700"

        # Format speed mode status
        speed_status = "ON" if self.speed_mode else "OFF"
        speed_color = "#FF4500" if self.speed_mode else "#FFFFFF"

        # Update display
        display_stats = {
            "Running": "Yes" if self.running else "No",
            "Population": f"{stats['population_density']*100:.1f}%",
            "Rate Change": f"{stats['population_change_rate']:.1f}",
            "Generation": str(stats["generation"]),
            "Cycle Status": cycle_status,
            "Speed Mode": speed_status,
            "FPS": f"{fps:.1f}",
        }

        colors = {"Cycle Status": cycle_color, "Speed Mode": speed_color}

        for stat, value in display_stats.items():
            color = colors.get(stat, "#FFFFFF")
            self.stats_labels[stat].config(text=f"{stat}: {value}", fg=color)

    def update_loop(self) -> None:
        """Main update loop."""
        current_time = self.master.tk.call("clock", "milliseconds")

        if self.speed_mode and self.running:
            # Speed mode: run many generations quickly
            generations_per_batch = 1000
            for _ in range(generations_per_batch):
                if not self.running:
                    break

                self.game.step()

                # Stop if cycle detected or max generations reached
                if (
                    self.game.cycle_detected
                    or self.game.generation >= self.max_generations
                ):
                    self.speed_mode = False
                    self.speed_btn.config(text="Speed Mode", bg="#800080")
                    self.running = False
                    self.redraw_all_cells()
                    break

            # Update display periodically in speed mode
            if not self.speed_mode or self.game.generation % 1000 == 0:
                self.draw_changed_cells()

        else:
            # Normal mode: update at specified frame rate
            if current_time - self.last_update >= self.update_interval:
                if self.running:
                    self.game.step()
                    self.draw_changed_cells()
                self.last_update = current_time

        # Update statistics less frequently
        if current_time % 100 < 20:
            self.update_statistics()

        # Schedule next update
        delay = 1 if (self.speed_mode and self.running) else 16
        self.master.after(delay, self.update_loop)

    def save_pattern(self) -> None:
        """Save the initial pattern."""
        if self.initial_state is None:
            messagebox.showwarning(
                "No Pattern", "No initial pattern to save. Reset first."
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Save Pattern",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self.pattern_library.storage_dir),
        )

        if filename:
            try:
                with open(filename, "w") as f:
                    json.dump(self.initial_state, f, indent=2)

                messagebox.showinfo(
                    "Saved", f"Pattern saved to {os.path.basename(filename)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save pattern: {str(e)}")

    def save_current_state(self) -> None:
        """Save the current state."""
        filename = filedialog.asksaveasfilename(
            title="Save Current State",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self.pattern_library.storage_dir),
        )

        if filename:
            try:
                current_state = self.game.save_state()
                with open(filename, "w") as f:
                    json.dump(current_state, f, indent=2)

                messagebox.showinfo(
                    "Saved", f"Current state saved to {os.path.basename(filename)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save current state: {str(e)}")

    def load_pattern(self) -> None:
        """Load a pattern from file."""
        filename = filedialog.askopenfilename(
            title="Load Pattern",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self.pattern_library.storage_dir),
        )

        if filename:
            try:
                with open(filename, "r") as f:
                    state = json.load(f)

                # Try to load as game state
                self.running = False
                self.game.load_state(state)
                self.initial_state = state
                self.redraw_all_cells()

                messagebox.showinfo(
                    "Loaded", f"Pattern loaded from {os.path.basename(filename)}"
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pattern: {str(e)}")


def main() -> None:
    """Main entry point for the Tkinter GUI."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("Installing scipy for optimized performance...")
        import subprocess

        subprocess.check_call(["pip", "install", "scipy"])

    root = tk.Tk()
    root.resizable(False, False)

    # Check for test mode
    import sys

    test_mode = "--test" in sys.argv

    app = TkinterGameOfLifeGUI(root)

    if test_mode:
        print("Running in test mode...")
        app.running = True

        def auto_exit() -> None:
            print(f"Test completed. Ran {app.game.generation} generations.")
            if app.game.cycle_detected:
                print(
                    f"Cycle detected: length {app.game.cycle_length} "
                    f"starting at generation {app.game.cycle_start_generation}"
                )
            root.quit()
            root.destroy()

        root.after(3000, auto_exit)

    root.mainloop()


if __name__ == "__main__":
    main()
