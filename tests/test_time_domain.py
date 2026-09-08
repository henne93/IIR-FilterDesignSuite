"""Tests for time_domain.py: source generation -> ideal/Q14 filtering (docs/CONCEPT.md §11.3-§11.5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from filters.lowpass import LowPassFilter
from time_domain import (
    INT16_MAX,
    INT16_MIN,
    compute,
    filter_ideal,
    filter_q14,
    from_q14_samples,
    n_samples_for,
    to_q14_samples,
)

FS = 13_333.0


# --- n_samples_for -------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_ms, fs, expected",
    [
        (50.0, 13_333.0, round(0.05 * 13_333.0)),
        (1_000.0, 8_000.0, 8_000),
        (0.001, 5_000.0, 1),  # rounds to 0 samples -> clamped to at least 1
    ],
)
def test_n_samples_for(duration_ms, fs, expected):
    assert n_samples_for(duration_ms, fs) == expected


# --- Q14 sample mapping (§11.4) --------------------------------------------------


def test_to_q14_samples_scales_by_full_scale():
    result = to_q14_samples(np.array([0.5, -0.5, 0.0]), full_scale=1000)
    np.testing.assert_array_equal(result, [500, -500, 0])


def test_to_q14_samples_clips_to_int16_range():
    result = to_q14_samples(np.array([2.0, -2.0]), full_scale=32767)
    assert result[0] == INT16_MAX
    assert result[1] == INT16_MIN  # -65534 clipped to -32768


def test_to_q14_and_from_q14_round_trip():
    normalized = np.array([0.5, -0.25, 0.0, 1.0])
    full_scale = 4  # chosen so every value maps to an exact integer
    q14 = to_q14_samples(normalized, full_scale)
    back = from_q14_samples(q14, full_scale)
    np.testing.assert_allclose(back, normalized)


# --- ideal filtering (§11.5) -----------------------------------------------------


def test_filter_ideal_with_no_filters_is_passthrough():
    source = np.array([1.0, 2.0, 3.0])
    result = filter_ideal(source, [])
    np.testing.assert_array_equal(result, source)


def test_filter_ideal_single_filter_matches_lfilter_directly():
    filt = LowPassFilter(fs=FS, fc=1_000.0)
    source = np.random.default_rng(0).normal(size=64)
    result = filter_ideal(source, [filt])
    b, a = filt.ideal_coefficients().as_ba()
    expected = lfilter(b, a, source)
    np.testing.assert_allclose(result, expected)


def test_filter_ideal_cascades_in_series():
    f1 = LowPassFilter(fs=FS, fc=2_000.0)
    f2 = LowPassFilter(fs=FS, fc=1_000.0)
    source = np.random.default_rng(1).normal(size=32)
    result = filter_ideal(source, [f1, f2])

    b1, a1 = f1.ideal_coefficients().as_ba()
    b2, a2 = f2.ideal_coefficients().as_ba()
    expected = lfilter(b2, a2, lfilter(b1, a1, source))
    np.testing.assert_allclose(result, expected)


# --- Q14 filtering (§11.5, needs the compiled native backend) -------------------


def test_filter_q14_cascades_in_series(native_backend):
    f1 = LowPassFilter(fs=FS, fc=2_000.0)
    f2 = LowPassFilter(fs=FS, fc=1_000.0)
    source = to_q14_samples(np.random.default_rng(2).normal(size=16) * 0.1, full_scale=32767).tolist()

    result = filter_q14(source, [f1, f2], native_backend)

    mid = native_backend.process_samples(f1.q14_coefficients(native_backend), source)
    expected = native_backend.process_samples(f2.q14_coefficients(native_backend), mid)
    assert result == expected


# --- compute() (full pipeline) ---------------------------------------------------


def test_compute_without_backend_leaves_q14_none():
    source = np.array([0.1, 0.2, 0.3, 0.4])
    result = compute(source, [], fs=FS, full_scale=32767, backend=None)
    assert result.q14 is None
    np.testing.assert_array_equal(result.ideal, source)
    assert len(result.time_ms) == len(source)
    np.testing.assert_allclose(np.diff(result.time_ms), 1000.0 / FS)


def test_compute_with_backend_produces_q14(native_backend):
    filt = LowPassFilter(fs=FS, fc=1_000.0)
    source = 0.1 * np.sin(2.0 * np.pi * 500.0 * np.arange(32) / FS)
    result = compute(source, [filt], fs=FS, full_scale=32767, backend=native_backend)
    assert result.q14 is not None
    assert len(result.q14) == len(source)
