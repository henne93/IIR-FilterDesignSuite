"""C Q14 coefficient/response accuracy vs. Python ideal coefficients.

Implements CONTRACTS.md §4 (scipy-vs-C comparison rules), §6 (frequency-grid
rules), and §11 (tolerances "must be calibrated empirically ... Phase 2 ...
and only then locked in as fixed regression thresholds").

Calibration method and findings (see PR/task report for the full numbers):

- Coefficient error (|ideal_f64 - dequantized_q14|, per coefficient) stays
  around 3e-5 absolute across all 4 filter types, the full fs sweep, and a
  1000-point linspace design sweep -- far tighter than the 1e-3 starting
  point CONTRACTS.md proposed. Locked at 1e-4 (~3x measured margin).

- Response (dB) error is dominated by an effect the contract's "starting
  point" tolerance didn't anticipate: deep in the stopband, the *ideal*
  float64 response reaches -150..-210 dB, but a 14-bit-quantized biquad has
  a real coefficient-quantization noise floor around -70..-100 dB -- so
  comparing dB error at those points measures "how deep can 14 bits of
  coefficient precision null a stopband" (a real, physically meaningful
  fixed-point limitation), not a defect. Left unguarded, single points hit
  >100 dB of "error" purely from comparing two near-zero numbers in log
  space. To keep the regression check meaningful, comparisons are gated to
  frequency points where the *ideal* magnitude is above -20 dB (i.e. within
  20 dB of passband gain) -- the region where a dB-domain comparison is
  actually informative.
  Even with that gate, CONTRACTS.md §11's own note holds: designs near the
  extreme edges of [100, fc_max(fs)] (particularly a low fc at a high fs)
  are measurably more sensitive. Two tolerance tiers are locked:
    - "interior" (excluding the outer 5% of the design range on each side):
      max observed ~0.25 dB (HP) -> locked at 0.5 dB.
    - "full range" (including the extreme edges): max observed ~3.65 dB
      (HP, fc=220 Hz at fs=40000 Hz) -> locked at 5.0 dB.

- PK (peaking EQ) shares the interior tolerance (measured ~0.02 dB, far
  inside the 0.5 dB LP/HP/BP/AP bound) but needs its own, looser full-range
  tolerance: PK's extra Q/gain dimensions compound with the same low-fc/
  high-fs edge sensitivity above, and its true domain-corner worst case
  (fc=100 Hz, fs=40000 Hz, Q=4.0, gain_db=-15 -- i.e. the deepest, narrowest
  cut at the lowest edge) measures ~8.66 dB -- confirmed the actual maximum
  by a finer grid search around that corner, not just the coarse sweep grid
  below. Locked at 12.0 dB (~1.4x measured margin, same ratio as LP/HP/BP/AP's
  own full-range calibration above).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

from filters.allpass import AllPassFilter
from filters.bandpass import BandPassFilter
from filters.base import fc_max
from filters.highpass import HighPassFilter
from filters.lowpass import LowPassFilter
from filters.peak import PeakFilter

FS_SWEEP = [5_000.0, 8_000.0, 13_333.0, 22_050.0, 40_000.0]

# Locked regression thresholds -- see module docstring for calibration.
COEFFICIENT_ERROR_TOL = 1e-4
RESPONSE_ERROR_INTERIOR_DB = 0.5
RESPONSE_ERROR_FULL_RANGE_DB = 5.0
RESPONSE_ERROR_FULL_RANGE_DB_PK = 12.0  # PK's own, looser full-range tolerance -- see module docstring
RESPONSE_MAGNITUDE_FLOOR_DB = -20.0  # gate: only compare where ideal |H| > this
EDGE_FRACTION = 0.05  # "interior" excludes this fraction of the range at each end

# Coefficient-accuracy sweep: linspace(100, fc_max(fs), 1000), per CONTRACTS.md §6.3.
COEFFICIENT_SWEEP_N = 1000
# Response-error sweep uses fewer design points per fs (still full fs coverage);
# the Bode grid itself is still the full 500-point log grid from CONTRACTS.md §6.
RESPONSE_SWEEP_N = 40


def _coeff_errors(ideal, q14_float) -> list[float]:
    return [abs(getattr(ideal, k) - getattr(q14_float, k)) for k in ("b0", "b1", "b2", "a1", "a2")]


def _response_error_db(ideal_coeffs, q14_coeffs, fs, freq_hz):
    w = 2.0 * np.pi * freq_hz / fs
    b_i, a_i = ideal_coeffs.as_ba()
    b_q, a_q = q14_coeffs.as_ba()
    _, h_i = freqz(b_i, a_i, worN=w)
    _, h_q = freqz(b_q, a_q, worN=w)
    mag_i = 20.0 * np.log10(np.abs(h_i))
    mag_q = 20.0 * np.log10(np.abs(h_q))
    return mag_i, np.abs(mag_i - mag_q)


def _bode_grid(fs: float) -> np.ndarray:
    f_max = min(0.7 * fs, fs / 2.0 - 1.0)
    return np.logspace(np.log10(10.0), np.log10(f_max), 500)


LP_HP_AP_CASES = [
    ("LP", LowPassFilter, {}),
    ("HP", HighPassFilter, {}),
    ("AP", AllPassFilter, {"Q": 0.7071}),
]


@pytest.mark.parametrize("name,cls,extra", LP_HP_AP_CASES, ids=[c[0] for c in LP_HP_AP_CASES])
def test_coefficient_sweep_lp_hp_ap(native_backend, name, cls, extra):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        for fc in np.linspace(100.0, hi, COEFFICIENT_SWEEP_N):
            f = cls(fs=fs, fc=fc, **extra)
            ideal = f.ideal_coefficients()
            q14_float = f.q14_coefficients(native_backend).to_float()
            max_err = max(max_err, max(_coeff_errors(ideal, q14_float)))
    assert max_err < COEFFICIENT_ERROR_TOL, f"{name}: max coefficient error {max_err:.3e}"


PK_Q_CASES = [0.8, 1.5, 4.0]
PK_GAIN_CASES = [-15.0, 0.0, 6.0, 15.0]


def test_coefficient_sweep_pk(native_backend):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        for fc in np.linspace(100.0, hi, COEFFICIENT_SWEEP_N // 50):
            for q in PK_Q_CASES:
                for gain_db in PK_GAIN_CASES:
                    f = PeakFilter(fs=fs, fc=fc, Q=q, gain_db=gain_db)
                    ideal = f.ideal_coefficients()
                    q14_float = f.q14_coefficients(native_backend).to_float()
                    max_err = max(max_err, max(_coeff_errors(ideal, q14_float)))
    assert max_err < COEFFICIENT_ERROR_TOL, f"PK: max coefficient error {max_err:.3e}"


def test_response_error_pk(native_backend):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        grid = _bode_grid(fs)
        for fc in np.linspace(100.0, hi, RESPONSE_SWEEP_N // 4):
            for q in PK_Q_CASES:
                for gain_db in PK_GAIN_CASES:
                    f = PeakFilter(fs=fs, fc=fc, Q=q, gain_db=gain_db)
                    ideal = f.ideal_coefficients()
                    q14_float = f.q14_coefficients(native_backend).to_float()
                    mag_i, err = _response_error_db(ideal, q14_float, fs, grid)
                    mask = mag_i > RESPONSE_MAGNITUDE_FLOOR_DB
                    if mask.any():
                        max_err = max(max_err, err[mask].max())
    assert max_err < RESPONSE_ERROR_FULL_RANGE_DB_PK, f"PK: max response error {max_err:.3f} dB"


def test_coefficient_sweep_bp(native_backend):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        for f_low in np.linspace(100.0, hi * 0.6, COEFFICIENT_SWEEP_N // 10):
            f_high = min(hi, f_low + (hi - f_low) * 0.5 + 50.0)
            if f_high <= f_low:
                continue
            f = BandPassFilter(fs=fs, f_low=f_low, f_high=f_high)
            ideal = f.ideal_coefficients()
            q14_float = f.q14_coefficients(native_backend).to_float()
            max_err = max(max_err, max(_coeff_errors(ideal, q14_float)))
    assert max_err < COEFFICIENT_ERROR_TOL, f"BP: max coefficient error {max_err:.3e}"


@pytest.mark.parametrize("name,cls,extra", LP_HP_AP_CASES, ids=[c[0] for c in LP_HP_AP_CASES])
def test_response_error_interior_lp_hp_ap(native_backend, name, cls, extra):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        lo_edge = 100.0 + EDGE_FRACTION * (hi - 100.0)
        hi_edge = hi - EDGE_FRACTION * (hi - 100.0)
        grid = _bode_grid(fs)
        for fc in np.linspace(lo_edge, hi_edge, RESPONSE_SWEEP_N):
            f = cls(fs=fs, fc=fc, **extra)
            ideal = f.ideal_coefficients()
            q14_float = f.q14_coefficients(native_backend).to_float()
            mag_i, err = _response_error_db(ideal, q14_float, fs, grid)
            mask = mag_i > RESPONSE_MAGNITUDE_FLOOR_DB
            if mask.any():
                max_err = max(max_err, err[mask].max())
    assert max_err < RESPONSE_ERROR_INTERIOR_DB, f"{name}: max interior response error {max_err:.3f} dB"


@pytest.mark.parametrize("name,cls,extra", LP_HP_AP_CASES, ids=[c[0] for c in LP_HP_AP_CASES])
def test_response_error_full_range_lp_hp_ap(native_backend, name, cls, extra):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        grid = _bode_grid(fs)
        for fc in np.linspace(100.0, hi, RESPONSE_SWEEP_N):
            f = cls(fs=fs, fc=fc, **extra)
            ideal = f.ideal_coefficients()
            q14_float = f.q14_coefficients(native_backend).to_float()
            mag_i, err = _response_error_db(ideal, q14_float, fs, grid)
            mask = mag_i > RESPONSE_MAGNITUDE_FLOOR_DB
            if mask.any():
                max_err = max(max_err, err[mask].max())
    assert max_err < RESPONSE_ERROR_FULL_RANGE_DB, f"{name}: max full-range response error {max_err:.3f} dB"


# Locked int16 headroom regression bound (CONTRACTS.md §7): a full-domain
# sweep (this test) measures a true worst case of ~32712 Q14 counts (AP
# a1/b1, near Q=4.0, fc near 100 Hz) -- comfortably under INT16_MAX (32767)
# but tight. PK's own worst case (b1/a1, near fc=100 Hz, Q=4.0, gain_db=+/-15)
# is ~32737 counts -- tighter than AP's but still under this bound. PK's
# Q_MIN is 0.8, not 0.25 like AP: at lower Q combined with +/-15 dB gain, PK's
# b0/b2 are NOT bounded by ~2.0 the way every other filter type's coefficients
# are (e.g. Q=0.25 at gain_db=+15 pushes b0 to ~3.11, real saturation, not
# quantization noise) -- see filters/peak.py's module docstring. This bound
# catches any future formula/range change that erodes that headroom before it
# becomes a silent int16 saturation in the field.
INT16_HEADROOM_BOUND = 32750
AP_Q_SWEEP = np.linspace(0.25, 4.0, 12)
PK_Q_SWEEP = np.linspace(0.8, 4.0, 8)
PK_GAIN_SWEEP = np.linspace(-15.0, 15.0, 5)


def test_no_coefficient_saturates_int16_across_domain(native_backend):
    """Regression guard for the int16_t q14_coeffs_t storage width
    (CONTRACTS.md §7): no in-domain LP/HP/BP/AP/PK design's quantized
    coefficient may approach INT16_MAX/INT16_MIN (32767/-32768)."""
    max_abs = 0

    def track(q14):
        nonlocal max_abs
        max_abs = max(max_abs, max(abs(getattr(q14, k)) for k in ("b0", "b1", "b2", "a1", "a2")))

    for fs in FS_SWEEP:
        hi = fc_max(fs)
        fcs = np.linspace(100.0, hi, 40)
        for fc in fcs:
            track(LowPassFilter(fs=fs, fc=fc).q14_coefficients(native_backend))
            track(HighPassFilter(fs=fs, fc=fc).q14_coefficients(native_backend))
            for q in AP_Q_SWEEP:
                track(AllPassFilter(fs=fs, fc=fc, Q=q).q14_coefficients(native_backend))
            for q in PK_Q_SWEEP:
                for gain_db in PK_GAIN_SWEEP:
                    track(PeakFilter(fs=fs, fc=fc, Q=q, gain_db=gain_db).q14_coefficients(native_backend))
        for f_low in np.linspace(100.0, hi * 0.9, 20):
            for f_high in np.linspace(f_low + 10.0, hi, 5):
                if f_high <= f_low:
                    continue
                track(BandPassFilter(fs=fs, f_low=f_low, f_high=f_high).q14_coefficients(native_backend))

    assert max_abs < INT16_HEADROOM_BOUND, f"max |Q14 coefficient| {max_abs} approached int16 saturation"


def test_response_error_bp(native_backend):
    max_err = 0.0
    for fs in FS_SWEEP:
        hi = fc_max(fs)
        grid = _bode_grid(fs)
        for f_low in np.linspace(100.0, hi * 0.6, RESPONSE_SWEEP_N // 2):
            f_high = min(hi, f_low + (hi - f_low) * 0.5 + 50.0)
            if f_high <= f_low:
                continue
            f = BandPassFilter(fs=fs, f_low=f_low, f_high=f_high)
            ideal = f.ideal_coefficients()
            q14_float = f.q14_coefficients(native_backend).to_float()
            mag_i, err = _response_error_db(ideal, q14_float, fs, grid)
            mask = mag_i > RESPONSE_MAGNITUDE_FLOOR_DB
            if mask.any():
                max_err = max(max_err, err[mask].max())
    assert max_err < RESPONSE_ERROR_INTERIOR_DB, f"BP: max response error {max_err:.3f} dB"
