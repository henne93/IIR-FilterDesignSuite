"""Time-domain plot widget (docs/CONCEPT.md §11 -- rough GUI pass, no wiring yet).

Single-axes matplotlib plot: source signal, ideal-filtered, and Q14-filtered
curves against time. Mirrors the placeholder/plot split used by
`ui/widgets/gain_widget.py` (`show_message()` for empty/placeholder states,
`plot()` for real data) so this slots into the same pattern once the signal
chain and filtering are wired up.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

SOURCE_COLOR = "#7f8c8d"
IDEAL_COLOR = "#2980b9"
Q14_COLOR = "#c0392b"


class TimeDomainWidget(QWidget):
    """Single-axis matplotlib plot embedded in a QWidget: source/ideal/Q14 vs. time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5.0, 3.0), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.subplots(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude (normalized)")
        self.ax.grid(True, alpha=0.3)

        self.show_message("Time-domain view -- concept skeleton, not yet wired up")

    def _clear_content(self) -> None:
        for artist in list(self.ax.lines) + list(self.ax.texts):
            artist.remove()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

    def plot(
        self,
        time_ms: np.ndarray,
        source: np.ndarray,
        ideal: np.ndarray | None,
        q14: np.ndarray | None,
    ) -> None:
        """Plots source (solid gray), ideal-filtered (solid blue), Q14-filtered (dashed red)."""
        self._clear_content()
        self.ax.plot(time_ms, source, color=SOURCE_COLOR, label="Source")
        if ideal is not None:
            self.ax.plot(time_ms, ideal, color=IDEAL_COLOR, label="Ideal filtered")
        if q14 is not None:
            self.ax.plot(time_ms, q14, color=Q14_COLOR, linestyle="--", label="Q14 filtered")
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(loc="best", fontsize="small")
        self.canvas.draw_idle()

    def show_message(self, message: str) -> None:
        """Clears axis content and shows a centered message (empty/placeholder states)."""
        self._clear_content()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes, wrap=True)
        self.canvas.draw_idle()
