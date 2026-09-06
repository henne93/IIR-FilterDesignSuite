"""Empirical, bit-exact stress test for biquad_q14.c's forced 32-bit
saturating accumulator (CONTRACTS.md §7).

The user explicitly chose a 32-bit accumulator over a 64-bit one after being
told the theoretical worst-case 5-term product sum (~5.4e9) is about 2.5x
over int32_t range, and that no formal per-filter-type stability bound rules
out reaching it. This file empirically probes that accepted risk: a
bit-exact Python mirror of the C saturating-add/subtract algorithm
(_python_reference_process below) is compared against the real compiled
biquad_q14_process() under adversarial full-scale alternating input, for a
representative LP/HP/BP/AP set including the domain's known-tightest corner
(AP, fc=100 Hz, fs=40000 Hz, Q=4.0 -- see test_native_coefficients.py's
int16-headroom regression test). Bit-exact equality (not just "doesn't
crash") proves the C saturation logic behaves exactly as designed; the
reported saturation-event count shows how often -- if ever -- the clamp is
actually reached for real coefficient magnitudes.
"""

from __future__ import annotations

import pytest

from filters.allpass import AllPassFilter
from filters.bandpass import BandPassFilter
from filters.base import Q14Coefficients
from filters.highpass import HighPassFilter
from filters.lowpass import LowPassFilter

INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

N_SAMPLES = 2000


def _sat_add(a: int, b: int) -> tuple[int, bool]:
    r = a + b
    if r > INT32_MAX or r < INT32_MIN:
        return (INT32_MAX if b > 0 else INT32_MIN), True
    return r, False


def _sat_sub(a: int, b: int) -> tuple[int, bool]:
    r = a - b
    if r > INT32_MAX or r < INT32_MIN:
        return (INT32_MAX if b < 0 else INT32_MIN), True
    return r, False


def _saturate_i16(v: int) -> int:
    return max(INT16_MIN, min(INT16_MAX, v))


def _python_reference_process(coeffs: Q14Coefficients, samples: list[int]) -> tuple[list[int], int]:
    """Bit-exact mirror of biquad_q14_process() (src/c/biquad_q14.c), using
    Python's arbitrary-precision ints to detect every saturating clamp.
    Returns (outputs, saturation_event_count)."""
    x1 = x2 = y1 = y2 = 0
    saturations = 0
    ys: list[int] = []
    for x in samples:
        p_b0 = coeffs.b0 * x
        p_b1 = coeffs.b1 * x1
        p_b2 = coeffs.b2 * x2
        p_a1 = coeffs.a1 * y1
        p_a2 = coeffs.a2 * y2

        acc = 0
        for p, op in ((p_b0, _sat_add), (p_b1, _sat_add), (p_b2, _sat_add), (p_a1, _sat_sub), (p_a2, _sat_sub)):
            acc, hit = op(acc, p)
            saturations += hit

        half = 1 << 13
        if acc >= 0:
            added, hit = _sat_add(acc, half)
            saturations += hit
            shifted = added >> 14
        else:
            neg = INT32_MAX if acc == INT32_MIN else -acc
            added, hit = _sat_add(neg, half)
            saturations += hit
            shifted = -(added >> 14)

        y = _saturate_i16(shifted)
        x2, x1 = x1, x
        y2, y1 = y1, y
        ys.append(y)
    return ys, saturations


CASES = [
    ("LP_low_fc_high_fs", lambda: LowPassFilter(fs=40_000.0, fc=100.0)),
    ("HP_low_fc_high_fs", lambda: HighPassFilter(fs=40_000.0, fc=100.0)),
    ("BP_wide_band", lambda: BandPassFilter(fs=40_000.0, f_low=100.0, f_high=18_000.0)),
    ("AP_worst_corner", lambda: AllPassFilter(fs=40_000.0, fc=100.0, Q=4.0)),
]


@pytest.mark.parametrize("name,make_filter", CASES, ids=[c[0] for c in CASES])
def test_saturating_accumulator_matches_python_reference_under_adversarial_input(native_backend, name, make_filter):
    coeffs = make_filter().q14_coefficients(native_backend)
    samples = [32767 if i % 2 == 0 else -32768 for i in range(N_SAMPLES)]

    y_c = native_backend.process_samples(coeffs, samples)
    y_ref, saturations = _python_reference_process(coeffs, samples)

    assert y_c == y_ref, f"{name}: C output diverges from the saturating-accumulator reference"
    print(f"\n{name}: {saturations} saturation event(s) across {N_SAMPLES} samples x 6 accumulation points")
