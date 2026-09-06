import cmath
import math

import numpy as np
import pytest

from conftest import edge_fcs
from filters.base import fc_max
from filters.highpass import HighPassFilter

MINUS_3_0103_DB = 20.0 * math.log10(1.0 / math.sqrt(2.0))  # exact Butterworth edge, CONTRACTS.md precision note


def _h_at(coeffs, w):
    z_inv = cmath.exp(-1j * w)
    num = coeffs.b0 + coeffs.b1 * z_inv + coeffs.b2 * z_inv**2
    den = 1.0 + coeffs.a1 * z_inv + coeffs.a2 * z_inv**2
    return num / den


def test_independent_complex_arithmetic_cross_check(fs):
    for fc in edge_fcs(fs):
        c = HighPassFilter(fs=fs, fc=fc).ideal_coefficients()
        w = 2.0 * math.pi * fc / fs
        h = _h_at(c, w)
        mag_db = 20.0 * math.log10(abs(h))
        assert mag_db == pytest.approx(MINUS_3_0103_DB, abs=1e-9)


@pytest.mark.parametrize("fc_frac", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_minus_3_0103_db_at_fc(fs, fc_frac):
    fc = fc_frac * fc_max(fs) + (1 - fc_frac) * 100.0
    filt = HighPassFilter(fs=fs, fc=fc)
    response = filt.ideal_response(np.array([fc]))
    assert response.magnitude_db[0] == pytest.approx(MINUS_3_0103_DB, abs=1e-6)


def test_minus_3db_across_full_domain(fs):
    for fc in edge_fcs(fs):
        filt = HighPassFilter(fs=fs, fc=fc)
        response = filt.ideal_response(np.array([fc]))
        assert response.magnitude_db[0] == pytest.approx(MINUS_3_0103_DB, abs=1e-6)


def test_zero_at_dc():
    # HP numerator (b0=b2=1/norm, b1=-2/norm) sums to exactly zero at DC.
    filt = HighPassFilter(fs=13333.0, fc=3000.0)
    c = filt.ideal_coefficients()
    assert c.b0 + c.b1 + c.b2 == pytest.approx(0.0, abs=1e-12)


def test_a1_a2_match_lowpass_same_fc_fs():
    from filters.lowpass import LowPassFilter

    lp = LowPassFilter(fs=13333.0, fc=3000.0).ideal_coefficients()
    hp = HighPassFilter(fs=13333.0, fc=3000.0).ideal_coefficients()
    assert hp.a1 == pytest.approx(lp.a1, abs=1e-12)
    assert hp.a2 == pytest.approx(lp.a2, abs=1e-12)


def test_derived_params_empty():
    filt = HighPassFilter(fs=13333.0, fc=3000.0)
    assert filt.derived_params() == {}


def test_kind():
    assert HighPassFilter.kind == "HP"


@pytest.mark.parametrize("bad_fc", [99.9, 0.0, -100.0])
def test_fc_below_minimum_raises(bad_fc):
    with pytest.raises(ValueError):
        HighPassFilter(fs=13333.0, fc=bad_fc)


def test_fc_above_fc_max_raises():
    fs = 13333.0
    with pytest.raises(ValueError):
        HighPassFilter(fs=fs, fc=fc_max(fs) + 1.0)


def test_q14_coefficients_dispatches_to_backend(fake_backend):
    filt = HighPassFilter(fs=13333.0, fc=3000.0)
    filt.q14_coefficients(fake_backend)
    assert fake_backend.calls == [("design_hp", (3000.0, 13333.0))]
