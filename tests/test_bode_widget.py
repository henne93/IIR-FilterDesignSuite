"""Unit tests for BodeWidget's adaptive magnitude y-axis helper.

Pure-function coverage only (no QWidget construction needed) -- the actual
`ax_mag.set_ylim(*_adaptive_magnitude_ylim(...))` wiring in `plot()` is
trivial glue over this. Mirrors the equivalent tests for the sibling copy in
`export.py` (`tests/test_export.py`'s "adaptive magnitude y-axis floor"
section) -- the two implementations are kept independent by design (see
`export.py`'s module docstring on UI/export layering) but must agree
behaviorally.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.widgets.bode_widget import MAGNITUDE_YLIM_FLOOR, MAGNITUDE_YLIM_STEP, _adaptive_magnitude_ylim


def test_adaptive_magnitude_ylim_keeps_top_unchanged():
    _bottom, top = _adaptive_magnitude_ylim(-37.2, 12.6)
    assert top == 12.6  # top bound formula is untouched by the adaptive-floor change


def test_adaptive_magnitude_ylim_rounds_bottom_down_to_clean_step():
    bottom, _top = _adaptive_magnitude_ylim(-37.2, 12.6)
    assert bottom == -40.0  # floor(-37.2 / 10) * 10
    assert bottom % MAGNITUDE_YLIM_STEP == 0


def test_adaptive_magnitude_ylim_caps_bottom_at_floor_for_deep_notches():
    # A filter with an extreme notch/null must not drag the whole axis down
    # past the -100 dB floor and squash everything else into a sliver.
    bottom, top = _adaptive_magnitude_ylim(-250.0, 3.0)
    assert bottom == MAGNITUDE_YLIM_FLOOR
    assert top == 3.0


def test_adaptive_magnitude_ylim_does_not_squash_small_excursion_filters():
    # A Peak/EQ filter with a small, mostly-flat response shouldn't be
    # forced down to the full -100 dB floor -- that's the point of the
    # adaptive bottom (CONTRACTS.md-adjacent UX request).
    bottom, top = _adaptive_magnitude_ylim(-2.1, 6.4)
    assert bottom > MAGNITUDE_YLIM_FLOOR
    assert bottom == -10.0
    assert top == 6.4


def test_adaptive_magnitude_ylim_guards_against_degenerate_span():
    bottom, top = _adaptive_magnitude_ylim(-150.0, -120.0)  # both below the floor
    assert top == MAGNITUDE_YLIM_FLOOR
    assert bottom == MAGNITUDE_YLIM_FLOOR - MAGNITUDE_YLIM_STEP
    assert bottom < top
