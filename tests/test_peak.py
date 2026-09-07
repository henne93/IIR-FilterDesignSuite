import cmath
import math

import numpy as np
import pytest

from filters.base import fc_max
from filters.peak import GAIN_MAX_DB, GAIN_MIN_DB, Q_MAX, Q_MIN, PeakFilter


def _h_at(coeffs, w):
    z_inv = cmath.exp(-1j * w)
    num = coeffs.b0 + coeffs.b1 * z_inv + coeffs.b2 * z_inv**2
    den = 1.0 + coeffs.a1 * z_inv + coeffs.a2 * z_inv**2
    return num / den


@pytest.mark.parametrize("q", [Q_MIN, 1.0, Q_MAX])
@pytest.mark.parametrize("gain_db", [GAIN_MIN_DB, -6.0, 0.0, 6.0, GAIN_MAX_DB])
def test_independent_complex_arithmetic_cross_check(fs, q, gain_db):
    """|H(fc)| == exactly gain_db dB; |H(DC)| == |H(Nyquist)| == exactly 0 dB."""
    fc = 0.4 * fc_max(fs) + 0.3 * 100.0
    c = PeakFilter(fs=fs, fc=fc, Q=q, gain_db=gain_db).ideal_coefficients()
    w0 = 2.0 * math.pi * fc / fs
    mag_at_fc_db = 20.0 * math.log10(abs(_h_at(c, w0)))
    assert mag_at_fc_db == pytest.approx(gain_db, abs=1e-6)

    dc_db = 20.0 * math.log10(abs(_h_at(c, 1e-9)))
    nyquist_db = 20.0 * math.log10(abs(_h_at(c, math.pi)))
    assert dc_db == pytest.approx(0.0, abs=1e-6)
    assert nyquist_db == pytest.approx(0.0, abs=1e-6)


def test_zero_gain_is_pure_passthrough(fs):
    """gain_db=0 (A=1) collapses to b0=1, b1=a1, b2=a2 -> H(z) == 1 everywhere."""
    fc = 0.4 * fc_max(fs) + 0.3 * 100.0
    filt = PeakFilter(fs=fs, fc=fc, Q=1.3, gain_db=0.0)
    freqs = np.logspace(np.log10(10.0), np.log10(fs / 2.0 - 1.0), 200)
    response = filt.ideal_response(freqs)
    assert response.magnitude_db == pytest.approx(0.0, abs=1e-9)
    assert response.phase_deg == pytest.approx(0.0, abs=1e-6)


def test_boost_and_cut_are_symmetric_in_db(fs):
    """+gain and -gain at the same fc/Q are mirror images in dB at fc."""
    fc = 0.4 * fc_max(fs) + 0.3 * 100.0
    boost = PeakFilter(fs=fs, fc=fc, Q=1.0, gain_db=9.0)
    cut = PeakFilter(fs=fs, fc=fc, Q=1.0, gain_db=-9.0)
    resp_boost = boost.ideal_response(np.array([fc]))
    resp_cut = cut.ideal_response(np.array([fc]))
    assert resp_boost.magnitude_db[0] == pytest.approx(9.0, abs=1e-6)
    assert resp_cut.magnitude_db[0] == pytest.approx(-9.0, abs=1e-6)


def test_default_q_is_valid():
    filt = PeakFilter(fs=13333.0, fc=3000.0, Q=1.0, gain_db=6.0)
    assert filt.Q == pytest.approx(1.0)


def test_derived_params_empty():
    filt = PeakFilter(fs=13333.0, fc=3000.0, Q=1.0, gain_db=6.0)
    assert filt.derived_params() == {}


def test_kind():
    assert PeakFilter.kind == "PK"


@pytest.mark.parametrize("bad_q", [0.79, 4.01, 0.25, 0.0, -1.0])
def test_q_out_of_range_raises(bad_q):
    with pytest.raises(ValueError):
        PeakFilter(fs=13333.0, fc=3000.0, Q=bad_q, gain_db=6.0)


def test_q_range_boundaries_are_valid():
    PeakFilter(fs=13333.0, fc=3000.0, Q=Q_MIN, gain_db=6.0)
    PeakFilter(fs=13333.0, fc=3000.0, Q=Q_MAX, gain_db=6.0)


@pytest.mark.parametrize("bad_gain", [-15.01, 15.01, -100.0, 100.0])
def test_gain_out_of_range_raises(bad_gain):
    with pytest.raises(ValueError):
        PeakFilter(fs=13333.0, fc=3000.0, Q=1.0, gain_db=bad_gain)


def test_gain_range_boundaries_are_valid():
    PeakFilter(fs=13333.0, fc=3000.0, Q=1.0, gain_db=-15.0)
    PeakFilter(fs=13333.0, fc=3000.0, Q=1.0, gain_db=15.0)


def test_fc_out_of_range_raises():
    fs = 13333.0
    with pytest.raises(ValueError):
        PeakFilter(fs=fs, fc=fc_max(fs) + 1.0, Q=1.0, gain_db=6.0)
    with pytest.raises(ValueError):
        PeakFilter(fs=fs, fc=99.0, Q=1.0, gain_db=6.0)


def test_q14_coefficients_dispatches_to_backend(fake_backend):
    filt = PeakFilter(fs=13333.0, fc=3000.0, Q=1.0, gain_db=6.0)
    filt.q14_coefficients(fake_backend)
    assert fake_backend.calls == [("design_pk", (3000.0, 13333.0, 1.0, 6.0))]
