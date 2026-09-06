"""Automated impulse-response test for biquad_q14.c's process() (CONTRACTS.md §7).

Response comparison elsewhere in this suite (test_native_coefficients.py)
stays entirely in the coefficient/freqz domain and never calls process() --
so a bug confined to process()'s delay-line/state-update logic (swapped
registers, an off-by-one in the shift, a saturation bug) would pass every
other test undetected. This file closes that gap by driving process()'s
actual state-update path: a Q14 unit impulse (amplitude = Q14 SCALE, i.e.
float 1.0) is fed through biquad_q14_process() sample-by-sample via ctypes,
and the resulting Q14 impulse response is compared against
scipy.signal.dimpulse() evaluated on the *dequantized* Q14 coefficients
(the coefficient-domain reference named in CONTRACTS.md §7), not the
float64 "ideal" coefficients -- process() only ever sees the quantized
values, so that's the correct ground truth for this specific check.

Calibration: across LP/HP/BP/AP and the full fs sweep, sampled at n=64
(chosen because for a lightly-damped 2nd-order design near the low end of
the supported fc range, the impulse response has a very slow decay -- pole
radius close to 1 -- so per-sample error from quantization-perturbed pole
angle keeps growing with n; n=64 captures the meaningful early transient
without running into that unbounded long-tail drift), the observed max
absolute error (in float, Q14-scale units) was ~0.0198, at an All-Pass
design with fc=100 Hz, fs=40000 Hz -- again the extreme low-fc/high-fs
corner of the design range. Locked at 0.05 (~2.5x measured margin).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import dimpulse

from filters.allpass import AllPassFilter
from filters.bandpass import BandPassFilter
from filters.base import Q14Coefficients, fc_max
from filters.highpass import HighPassFilter
from filters.lowpass import LowPassFilter

FS_SWEEP = [5_000.0, 8_000.0, 13_333.0, 22_050.0, 40_000.0]
N_SAMPLES = 64
IMPULSE_ERROR_TOL = 0.05

LP_HP_AP_CASES = [
    ("LP", LowPassFilter, {}),
    ("HP", HighPassFilter, {}),
    ("AP", AllPassFilter, {"Q": 0.7071}),
]


def _impulse_reference(q14_coeffs: Q14Coefficients, n: int) -> np.ndarray:
    b, a = q14_coeffs.to_float().as_ba()
    _, (y,) = dimpulse((b, a, 1), n=n)
    return y.flatten()


def _impulse_error(native_backend, q14_coeffs: Q14Coefficients, n: int) -> np.ndarray:
    y_ref = _impulse_reference(q14_coeffs, n)
    y_c = native_backend.process_impulse(q14_coeffs, n)
    y_c_float = np.asarray(y_c, dtype=np.float64) / Q14Coefficients.SCALE
    return np.abs(y_ref - y_c_float)


@pytest.mark.parametrize("name,cls,extra", LP_HP_AP_CASES, ids=[c[0] for c in LP_HP_AP_CASES])
def test_impulse_response_matches_dequantized_reference_lp_hp_ap(native_backend, name, cls, extra):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        for fc in np.linspace(100.0, hi, 5):
            f = cls(fs=fs, fc=fc, **extra)
            q14_coeffs = f.q14_coefficients(native_backend)
            err = _impulse_error(native_backend, q14_coeffs, N_SAMPLES)
            max_err = max(max_err, err.max())
    assert max_err < IMPULSE_ERROR_TOL, f"{name}: max impulse-response error {max_err:.4f}"


def test_impulse_response_matches_dequantized_reference_bp(native_backend):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        for f_low in np.linspace(100.0, hi * 0.6, 5):
            f_high = min(hi, f_low + (hi - f_low) * 0.5 + 50.0)
            if f_high <= f_low:
                continue
            f = BandPassFilter(fs=fs, f_low=f_low, f_high=f_high)
            q14_coeffs = f.q14_coefficients(native_backend)
            err = _impulse_error(native_backend, q14_coeffs, N_SAMPLES)
            max_err = max(max_err, err.max())
    assert max_err < IMPULSE_ERROR_TOL, f"BP: max impulse-response error {max_err:.4f}"


def test_process_is_stateful_across_calls(native_backend):
    """Sanity check that state actually persists between process() calls
    (i.e. it isn't silently a stateless passthrough), independent of the
    dimpulse comparison above."""
    f = LowPassFilter(fs=13333.0, fc=3000.0)
    q14_coeffs = f.q14_coefficients(native_backend)
    y = native_backend.process_impulse(q14_coeffs, 8)
    assert any(sample != 0 for sample in y[1:]), "expected a non-trivial IIR tail after the impulse"
