"""Shared hover measurement cursor, scoped to one inspector tab.

Synchronizes a vertical frequency line across one `BodeWidget` and one
`GainWidget` on plain mouse movement (`motion_notify_event`) -- no click
required. No filter math or Q14 computation happens here: values come from
the `FrequencyResponse` pair a caller-supplied `response_provider()` hands
back, read via `numpy.interp` at the hovered frequency.

One `MeasurementCursor` per tab, connected once in that tab panel's
`__init__` -- `response_provider` is a closure over the panel's current
live data (e.g. `lambda: self._current_responses`), so a live-data refresh
never needs to reconnect matplotlib's event handlers (which would leak
duplicate connections); the callback simply reads fresh data next time the
mouse moves. Because each panel owns its own `MeasurementCursor` instance,
cursor state never leaks between tabs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from filters.base import FrequencyResponse
from ui.widgets.bode_widget import BodeWidget
from ui.widgets.gain_widget import GainWidget

ResponsePair = "tuple[FrequencyResponse, FrequencyResponse | None]"


class MeasurementCursor:
    """Keeps a `BodeWidget` and a `GainWidget`'s hover cursors in sync."""

    def __init__(
        self,
        bode: BodeWidget,
        gain: GainWidget,
        response_provider: "Callable[[], ResponsePair | None]",
    ) -> None:
        self.bode = bode
        self.gain = gain
        self._response_provider = response_provider
        bode.canvas.mpl_connect("motion_notify_event", self._on_bode_motion)
        gain.canvas.mpl_connect("motion_notify_event", self._on_gain_motion)

    def _on_bode_motion(self, event) -> None:
        if event.inaxes not in (self.bode.ax_mag, self.bode.ax_phase):
            return
        self._update(event.xdata)

    def _on_gain_motion(self, event) -> None:
        if event.inaxes is not self.gain.ax_gain:
            return
        self._update(event.xdata)

    def _update(self, freq_hz: float | None) -> None:
        if freq_hz is None or freq_hz <= 0:
            return
        pair = self._response_provider()
        if pair is None:
            return
        ideal, q14 = pair

        ideal_mag_db = float(np.interp(freq_hz, ideal.freq_hz, ideal.magnitude_db))
        ideal_phase_deg = float(np.interp(freq_hz, ideal.freq_hz, ideal.phase_deg))
        ideal_gain = 10.0 ** (ideal_mag_db / 20.0)

        q14_mag_db = q14_phase_deg = q14_gain = None
        if q14 is not None:
            q14_mag_db = float(np.interp(freq_hz, q14.freq_hz, q14.magnitude_db))
            q14_phase_deg = float(np.interp(freq_hz, q14.freq_hz, q14.phase_deg))
            q14_gain = 10.0 ** (q14_mag_db / 20.0)

        self.bode.set_cursor(freq_hz, ideal_mag_db, ideal_phase_deg, q14_mag_db, q14_phase_deg)
        self.gain.set_cursor(freq_hz, ideal_gain, q14_gain)
