import cmath
import math

import numpy as np
import pytest

from filters.bandpass import BandPassFilter
from filters.base import fc_max

MINUS_3_0103_DB = 20.0 * math.log10(1.0 / math.sqrt(2.0))  # exact Butterworth edge, CONTRACTS.md precision note


def _h_at(coeffs, w):
    z_inv = cmath.exp(-1j * w)
    num = coeffs.b0 + coeffs.b1 * z_inv + coeffs.b2 * z_inv**2
    den = 1.0 + coeffs.a1 * z_inv + coeffs.a2 * z_inv**2
    return num / den


def _bands(fs):
    """Bands from a narrow (~1-octave) width up to nearly the full valid range."""
    hi = fc_max(fs)
    return [
        (100.0, 200.0),
        (1_000.0, 2_000.0),
        (500.0, hi * 0.5),
        (200.0, hi),
        (hi * 0.3, hi),
    ]


def test_minus_3_0103_db_at_both_edges(fs):
    for f_low, f_high in _bands(fs):
        filt = BandPassFilter(fs=fs, f_low=f_low, f_high=f_high)
        response = filt.ideal_response(np.array([f_low, f_high]))
        assert response.magnitude_db[0] == pytest.approx(MINUS_3_0103_DB, abs=1e-6)
        assert response.magnitude_db[1] == pytest.approx(MINUS_3_0103_DB, abs=1e-6)


def test_independent_complex_arithmetic_cross_check(fs):
    for f_low, f_high in _bands(fs):
        c = BandPassFilter(fs=fs, f_low=f_low, f_high=f_high).ideal_coefficients()
        for f in (f_low, f_high):
            w = 2.0 * math.pi * f / fs
            h = _h_at(c, w)
            mag_db = 20.0 * math.log10(abs(h))
            assert mag_db == pytest.approx(MINUS_3_0103_DB, abs=1e-9)


def test_b1_always_zero(fs):
    for f_low, f_high in _bands(fs):
        c = BandPassFilter(fs=fs, f_low=f_low, f_high=f_high).ideal_coefficients()
        assert c.b1 == 0.0


def test_derived_fc_and_q():
    f_low, f_high = 2_000.0, 4_000.0
    filt = BandPassFilter(fs=13333.0, f_low=f_low, f_high=f_high)
    derived = filt.derived_params()
    expected_fc = math.sqrt(f_low * f_high)
    expected_q = expected_fc / (f_high - f_low)
    assert derived["fc"] == pytest.approx(expected_fc)
    assert derived["Q"] == pytest.approx(expected_q)
    assert set(derived.keys()) == {"fc", "Q"}


def test_kind():
    assert BandPassFilter.kind == "BP"


def test_f_low_ge_f_high_raises():
    with pytest.raises(ValueError):
        BandPassFilter(fs=13333.0, f_low=4000.0, f_high=2000.0)
    with pytest.raises(ValueError):
        BandPassFilter(fs=13333.0, f_low=3000.0, f_high=3000.0)


def test_f_low_below_minimum_raises():
    with pytest.raises(ValueError):
        BandPassFilter(fs=13333.0, f_low=50.0, f_high=2000.0)


def test_f_high_above_fc_max_raises():
    fs = 13333.0
    with pytest.raises(ValueError):
        BandPassFilter(fs=fs, f_low=1000.0, f_high=fc_max(fs) + 1.0)


def test_q14_coefficients_dispatches_to_backend(fake_backend):
    filt = BandPassFilter(fs=13333.0, f_low=2000.0, f_high=4000.0)
    filt.q14_coefficients(fake_backend)
    assert fake_backend.calls == [("design_bp", (2000.0, 4000.0, 13333.0))]
