"""Tests for error_analysis.py (Phase 3): CONCEPT.md §6, CONTRACTS.md §4/§6/§11/§12.

Numeric assertions reuse the locked Phase 2 tolerances (CONTRACTS.md §11) as
sanity bounds -- error_analysis.py is a thin, non-duplicating wrapper around
the same ideal_coefficients()/q14_coefficients()/freqz path already exercised
in tests/test_native_coefficients.py, so results should land in the same
range, not introduce new numerics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

from error_analysis import (
    BODE_GRID_N,
    COEFFICIENT_SWEEP_N,
    RESPONSE_MAGNITUDE_FLOOR_DB,
    CoefficientSweepError,
    ResponseError,
    bode_grid,
    coefficient_sweep,
    combined_response_error,
    response_error,
)
from filters.bandpass import BandPassFilter
from filters.base import fc_max
from filters.highpass import HighPassFilter
from filters.lowpass import LowPassFilter

# Locked Phase 2 tolerances (CONTRACTS.md §11), reused here as sanity bounds.
COEFFICIENT_ERROR_TOL = 1e-4
RESPONSE_ERROR_FULL_RANGE_DB = 5.0


# --- bode_grid ---------------------------------------------------------


def test_bode_grid_shape_and_bounds():
    fs = 13333.0
    grid = bode_grid(fs)
    assert grid.shape == (BODE_GRID_N,)
    assert grid[0] == pytest.approx(10.0)
    assert grid[-1] == pytest.approx(min(0.7 * fs, fs / 2.0 - 1.0))
    assert np.all(np.diff(grid) > 0)


def test_bode_grid_custom_n():
    grid = bode_grid(13333.0, n=50)
    assert grid.shape == (50,)


def test_bode_grid_rejects_nonpositive_fs():
    with pytest.raises(ValueError):
        bode_grid(0.0)
    with pytest.raises(ValueError):
        bode_grid(-100.0)


def test_bode_grid_rejects_fs_too_small_for_floor():
    # 0.7*fs <= 10 Hz for any fs <= ~14.3 Hz -- far below the supported
    # [5000, 40000] fs range, but bode_grid() should still fail explicitly
    # rather than emit an empty/inverted grid.
    with pytest.raises(ValueError):
        bode_grid(10.0)


# --- response_error ------------------------------------------------------


def test_response_error_missing_backend_raises():
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    with pytest.raises(ValueError, match="NativeBackend"):
        response_error(filt, None)


def test_response_error_matches_manual_freqz(native_backend):
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    result = response_error(filt, native_backend)
    assert isinstance(result, ResponseError)

    grid = bode_grid(filt.fs)
    ideal = filt.ideal_coefficients()
    q14_float = filt.q14_coefficients(native_backend).to_float()
    w = 2.0 * np.pi * grid / filt.fs
    _, h_i = freqz(*ideal.as_ba(), worN=w)
    _, h_q = freqz(*q14_float.as_ba(), worN=w)
    mag_i = 20.0 * np.log10(np.abs(h_i))
    mag_q = 20.0 * np.log10(np.abs(h_q))
    mask = mag_i > RESPONSE_MAGNITUDE_FLOOR_DB
    expected_err = np.abs(mag_i - mag_q)[mask]

    assert result.max_db == pytest.approx(expected_err.max())
    assert result.rms_db == pytest.approx(np.sqrt(np.mean(expected_err**2)))
    assert result.max_db < RESPONSE_ERROR_FULL_RANGE_DB


def test_response_error_gates_deep_stopband(native_backend):
    # A steep-rolloff LP at a low fc relative to fs pushes much of the Bode
    # grid deep into the stopband -- the gate should shrink the compared set.
    filt = LowPassFilter(fs=40_000.0, fc=200.0)
    result = response_error(filt, native_backend)
    assert result.freq_hz.shape[0] < BODE_GRID_N
    assert result.error_db.shape == result.freq_hz.shape
    assert np.all(result.error_db >= 0.0)


def test_response_error_rms_never_exceeds_max(native_backend):
    filt = HighPassFilter(fs=22_050.0, fc=5_000.0)
    result = response_error(filt, native_backend)
    assert result.rms_db <= result.max_db


def test_response_error_custom_grid(native_backend):
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    grid = np.linspace(100.0, 4000.0, 25)
    result = response_error(filt, native_backend, freq_hz=grid)
    assert result.freq_hz.shape[0] <= 25


# --- combined_response_error ----------------------------------------------


def test_combined_response_error_empty_chain_raises(native_backend):
    with pytest.raises(ValueError, match="empty"):
        combined_response_error([], native_backend)


def test_combined_response_error_missing_backend_raises():
    chain = [LowPassFilter(fs=13333.0, fc=3000.0)]
    with pytest.raises(ValueError, match="NativeBackend"):
        combined_response_error(chain, None)


def test_combined_response_error_rejects_mismatched_fs(native_backend):
    chain = [LowPassFilter(fs=13333.0, fc=3000.0), HighPassFilter(fs=22050.0, fc=1000.0)]
    with pytest.raises(ValueError, match="fs"):
        combined_response_error(chain, native_backend)


def test_combined_response_error_single_block_matches_response_error(native_backend):
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    single = response_error(filt, native_backend)
    combined = combined_response_error([filt], native_backend)
    assert combined.max_db == pytest.approx(single.max_db)
    assert combined.rms_db == pytest.approx(single.rms_db)


def test_combined_response_error_two_stage_cascade(native_backend):
    fs = 13333.0
    chain = [LowPassFilter(fs=fs, fc=4000.0), HighPassFilter(fs=fs, fc=1000.0)]
    result = combined_response_error(chain, native_backend)
    assert isinstance(result, ResponseError)
    assert result.max_db < RESPONSE_ERROR_FULL_RANGE_DB
    assert result.rms_db <= result.max_db


# --- coefficient_sweep -----------------------------------------------------


def test_coefficient_sweep_missing_backend_raises():
    with pytest.raises(ValueError, match="NativeBackend"):
        coefficient_sweep(13333.0, lambda fc: LowPassFilter(fs=13333.0, fc=fc), None)


def test_coefficient_sweep_lp(native_backend):
    fs = 13333.0
    result = coefficient_sweep(fs, lambda fc: LowPassFilter(fs=fs, fc=fc), native_backend)
    assert isinstance(result, CoefficientSweepError)
    assert result.max_abs < COEFFICIENT_ERROR_TOL
    assert result.rms_abs <= result.max_abs
    assert set(result.per_coefficient_max) == {"b0", "b1", "b2", "a1", "a2"}
    assert all(v <= result.max_abs for v in result.per_coefficient_max.values())
    assert 100.0 <= result.worst_frequency_hz <= fc_max(fs)


def test_coefficient_sweep_uses_full_locked_point_count(native_backend):
    fs = 13333.0
    calls = []
    original = LowPassFilter

    def counting_factory(fc):
        calls.append(fc)
        return original(fs=fs, fc=fc)

    coefficient_sweep(fs, counting_factory, native_backend)
    assert len(calls) == COEFFICIENT_SWEEP_N


def test_coefficient_sweep_custom_n(native_backend):
    fs = 13333.0
    result = coefficient_sweep(fs, lambda fc: LowPassFilter(fs=fs, fc=fc), native_backend, n=10)
    assert result.max_abs < COEFFICIENT_ERROR_TOL


def test_coefficient_sweep_rejects_too_few_points(native_backend):
    fs = 13333.0
    with pytest.raises(ValueError):
        coefficient_sweep(fs, lambda fc: LowPassFilter(fs=fs, fc=fc), native_backend, n=1)


def test_coefficient_sweep_bp_holds_bandwidth_fixed(native_backend):
    fs = 13333.0
    hi = fc_max(fs)
    half_bw = 0.02 * (hi - 100.0)  # narrow enough to stay in range at both sweep ends

    def design_at(center: float) -> BandPassFilter:
        f_low = max(100.0, center - half_bw)
        f_high = min(hi, center + half_bw)
        return BandPassFilter(fs=fs, f_low=f_low, f_high=f_high)

    result = coefficient_sweep(fs, design_at, native_backend)
    assert result.max_abs < COEFFICIENT_ERROR_TOL
    assert 100.0 <= result.worst_frequency_hz <= hi
