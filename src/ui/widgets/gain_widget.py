"""Linear-frequency, linear-gain plot widget (CONTRACTS.md §6, extended).

Sits below the Bode plot in every inspector tab. Plots ideal/Q14 *gain*
(`10 ** (magnitude_db / 20)`, converted at this plotting boundary only) on
linear x/y axes, over the same practical range as `bode_grid()` but linearly
spaced (`error_analysis.linear_response_grid()`). No filter equations or
Q14 computation happen here -- callers pass in `FrequencyResponse` objects
already computed by `FilterDesign`/`FilterChain`.

Shares the hover measurement cursor with `BodeWidget` via
`ui/widgets/measurement_cursor.py`; `set_cursor()`'s vertical line is added
as a `LineCollection` via `bode_widget.draw_cursor_line()` so it lands in
`ax.collections`, not `ax.lines`, keeping curve-count assertions
(`len(ax_gain.lines)`) accurate regardless of cursor state, and with
`autolim=False` so it never feeds into the axes' autoscale/data limits.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from filters.base import FrequencyResponse
from ui.widgets.bode_widget import IDEAL_COLOR, Q14_COLOR, draw_cursor_line


def _gain(response: FrequencyResponse):
    return 10.0 ** (response.magnitude_db / 20.0)


class GainWidget(QWidget):
    """Single-axis linear-frequency/linear-gain matplotlib plot embedded in a QWidget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5.0, 2.0), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_gain = self.figure.subplots(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax_gain.set_xlabel("Frequency (Hz)")
        self.ax_gain.set_ylabel("Gain (linear)")
        self.ax_gain.grid(True, alpha=0.3)

        self._cursor_artists: list = []

    def _clear_content(self) -> None:
        # Cursor artists must go first: the cursor annotation is a `Text`
        # that also lives in `ax_gain.texts` (see `BodeWidget._clear_content`
        # for why removing it twice raises `ValueError`).
        self._clear_cursor()
        for artist in list(self.ax_gain.lines) + list(self.ax_gain.texts):
            artist.remove()
        legend = self.ax_gain.get_legend()
        if legend is not None:
            legend.remove()

    def plot(self, ideal: FrequencyResponse, q14: FrequencyResponse | None) -> None:
        """Plots ideal (solid) and, if given, Q14 (dashed) linear-gain curves."""
        self._clear_content()
        self.ax_gain.plot(ideal.freq_hz, _gain(ideal), color=IDEAL_COLOR, label="Ideal")
        if q14 is not None:
            self.ax_gain.plot(q14.freq_hz, _gain(q14), color=Q14_COLOR, linestyle="--", label="Q14")
        self.ax_gain.relim()
        self.ax_gain.autoscale_view()
        self.ax_gain.legend(loc="best", fontsize="small")
        self.canvas.draw_idle()

    def show_message(self, message: str) -> None:
        """Clears axis content and shows a centered message (empty/invalid states)."""
        self._clear_content()
        self.ax_gain.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax_gain.transAxes, wrap=True)
        self.canvas.draw_idle()

    # -- measurement cursor (ui/widgets/measurement_cursor.py) --------------

    def set_cursor(self, freq_hz: float, ideal_gain: float, q14_gain: float | None) -> None:
        """Draws/updates the shared vertical measurement line and its annotation."""
        self._clear_cursor()
        self._cursor_artists.append(draw_cursor_line(self.ax_gain, freq_hz))
        label = f"{freq_hz:.1f} Hz\nIdeal gain: {ideal_gain:.4f}"
        if q14_gain is not None:
            label += f"\nQ14 gain: {q14_gain:.4f}"
        self._cursor_artists.append(
            self.ax_gain.annotate(
                label,
                xy=(0.02, 0.02),
                xycoords="axes fraction",
                fontsize="small",
                va="bottom",
                ha="left",
            )
        )
        self.canvas.draw_idle()

    def clear_cursor(self) -> None:
        self._clear_cursor()
        self.canvas.draw_idle()

    def _clear_cursor(self) -> None:
        for artist in self._cursor_artists:
            artist.remove()
        self._cursor_artists = []
