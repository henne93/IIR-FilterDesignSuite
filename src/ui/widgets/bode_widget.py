"""Stacked ideal/Q14 Bode plot widget (Phase 5 inspector, CONTRACTS.md §6).

Renders amplitude (dB) and unwrapped phase (deg) on two vertically stacked,
shared-x, log-frequency matplotlib axes. Takes `FrequencyResponse` objects
computed elsewhere (`FilterDesign.ideal_response`/`.q14_response`,
`FilterChain.combined_*_response`) -- no response math lives here.

`set_cursor()`/`clear_cursor()` draw the shared hover measurement line (see
`ui/widgets/measurement_cursor.py`); the vertical line is added as a
`LineCollection` via `draw_cursor_line()` below (not `Axes.plot()`), so it
lands in `ax.collections`, not `ax.lines` -- existing/new tests that count
response curves via `len(ax_mag.lines)` stay accurate whether or not a
cursor is currently drawn. It's added with `autolim=False` rather than via
`Axes.vlines()` so it never feeds into the axes' autoscale/data limits (see
`draw_cursor_line()`). `GainWidget` (`ui/widgets/gain_widget.py`) shares
this same helper for its own cursor line.
"""

from __future__ import annotations

import matplotlib.collections as mcollections
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from filters.base import FrequencyResponse

IDEAL_COLOR = "#2980b9"
Q14_COLOR = "#c0392b"
CURSOR_COLOR = "#555555"


def draw_cursor_line(ax, freq_hz: float) -> mcollections.LineCollection:
    """Draws a vertical measurement-cursor line spanning `ax`'s current y-range.

    Shared by `BodeWidget` and `GainWidget` (`ui/widgets/gain_widget.py`).
    Built directly as a `LineCollection` (what `Axes.vlines()` returns)
    rather than via `Axes.vlines()` itself: that helper always calls
    `update_datalim()` + `_request_autoscale_view()`, which would fold the
    line's y-extent (== the current ylim) into the axes' data limits. Since
    every redraw then re-pads that range with autoscale's default margin,
    each cursor update would nudge the y-axis outward -- a runaway feedback
    loop on every mouse move. `add_collection(..., autolim=False)` draws the
    same line without touching the axes' autoscale state.
    """
    ymin, ymax = ax.get_ylim()
    line = mcollections.LineCollection(
        [[(freq_hz, ymin), (freq_hz, ymax)]],
        colors=CURSOR_COLOR,
        linewidths=0.8,
        linestyles=":",
    )
    ax.add_collection(line, autolim=False)
    return line


class BodeWidget(QWidget):
    """Two-axis (magnitude, phase) matplotlib Bode plot embedded in a QWidget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5.0, 4.0), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_mag, self.ax_phase = self.figure.subplots(2, 1, sharex=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Scale/labels/grid are static axis configuration, set once here.
        # `plot()`/`show_message()` only add/remove plotted *content*
        # (`_clear_content()`) rather than calling `Axes.clear()` on every
        # redraw -- re-clearing already-log-scaled axes and immediately
        # re-applying `set_xscale("log")` is what triggers matplotlib's
        # "non-positive xlim on a log-scaled axis" warning on the second
        # and later redraws (the axes briefly see clear()'s default (0, 1)
        # linear-scale xlim before the log scale is reapplied).
        self.ax_mag.set_xscale("log")
        self.ax_phase.set_xscale("log")
        self.ax_mag.set_xlim(10.0, 20_000.0)  # sane default before any data is plotted
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True, which="both", alpha=0.3)
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.grid(True, which="both", alpha=0.3)

        self._cursor_artists: list = []

    def _clear_content(self) -> None:
        # Cursor artists must go first: the cursor annotation is a `Text`
        # that also lives in `ax.texts`, so clearing it here first (and
        # resetting `_cursor_artists`) keeps the loop below from trying to
        # remove the same artist a second time (`ValueError: list.remove(x):
        # x not in list`).
        self._clear_cursor()
        for ax in (self.ax_mag, self.ax_phase):
            for artist in list(ax.lines) + list(ax.texts):
                artist.remove()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    def plot(self, ideal: FrequencyResponse, q14: FrequencyResponse | None) -> None:
        """Plots ideal (solid) and, if given, Q14 (dashed) amplitude/phase curves."""
        self._clear_content()
        self.ax_mag.plot(ideal.freq_hz, ideal.magnitude_db, color=IDEAL_COLOR, label="Ideal")
        self.ax_phase.plot(ideal.freq_hz, ideal.phase_deg, color=IDEAL_COLOR, label="Ideal")
        if q14 is not None:
            self.ax_mag.plot(q14.freq_hz, q14.magnitude_db, color=Q14_COLOR, linestyle="--", label="Q14")
            self.ax_phase.plot(q14.freq_hz, q14.phase_deg, color=Q14_COLOR, linestyle="--", label="Q14")
        self.ax_mag.relim()
        self.ax_mag.autoscale_view()
        self.ax_phase.relim()
        self.ax_phase.autoscale_view()
        self.ax_mag.legend(loc="best", fontsize="small")
        self.canvas.draw_idle()

    def show_message(self, message: str) -> None:
        """Clears both axes' content and shows a centered message (empty/invalid states)."""
        self._clear_content()
        self.ax_mag.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax_mag.transAxes, wrap=True)
        self.canvas.draw_idle()

    # -- measurement cursor (ui/widgets/measurement_cursor.py) --------------

    def set_cursor(
        self,
        freq_hz: float,
        ideal_mag_db: float,
        ideal_phase_deg: float,
        q14_mag_db: float | None,
        q14_phase_deg: float | None,
    ) -> None:
        """Draws/updates the shared vertical measurement line and its annotation."""
        self._clear_cursor()
        self._cursor_artists.append(draw_cursor_line(self.ax_mag, freq_hz))
        self._cursor_artists.append(draw_cursor_line(self.ax_phase, freq_hz))
        label = f"{freq_hz:.1f} Hz\nIdeal: {ideal_mag_db:.2f} dB, {ideal_phase_deg:.1f}°"
        if q14_mag_db is not None:
            label += f"\nQ14: {q14_mag_db:.2f} dB, {q14_phase_deg:.1f}°"
        self._cursor_artists.append(
            self.ax_mag.annotate(
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
