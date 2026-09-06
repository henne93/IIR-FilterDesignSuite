import cmath
import math

import numpy as np
import pytest

from filters.base import fc_max
from filters.lowpass import LowPassFilter

from conftest import edge_fcs

MINUS_3_0103_DB = 20.0 * math.log10(1.0 / math.sqrt(2.0))  # exact Butterworth edge, CONTRACTS.md precision note


def _h_at(coeffs, w):
    """Independent (non-scipy) complex-arithmetic evaluation of H(e^jw).

    Cross-checks the freqz-based ideal_response() against a hand-rolled
    evaluation, per CONTRACTS.md §3's stated verification method.
    """
    z_inv = cmath.exp(-1j * w)
    num = coeffs.b0 + coeffs.b1 * z_inv + coeffs.b2 * z_inv**2
    den = 1.0 + coeffs.a1 * z_inv + coeffs.a2 * z_inv**2
    return num / den


@pytest.mark.parametrize("fc_frac", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_minus_3_0103_db_at_fc(fs, fc_frac):
    fc = fc_frac * fc_max(fs) + (1 - fc_frac) * 100.0
    filt = LowPassFilter(fs=fs, fc=fc)
    response = filt.ideal_response(np.array([fc]))
    assert response.magnitude_db[0] == pytest.approx(MINUS_3_0103_DB, abs=1e-6)


def test_minus_3db_across_full_domain(fs):
    for fc in edge_fcs(fs):
        filt = LowPassFilter(fs=fs, fc=fc)
        response = filt.ideal_response(np.array([fc]))
        assert response.magnitude_db[0] == pytest.approx(MINUS_3_0103_DB, abs=1e-6)


def test_independent_complex_arithmetic_cross_check(fs):
    # CONTRACTS.md §3 "Verification method": cross-check freqz against a
    # standalone (non-scipy) complex-arithmetic evaluation of H(e^jw) at fc.
    for fc in edge_fcs(fs):
        c = LowPassFilter(fs=fs, fc=fc).ideal_coefficients()
        w = 2.0 * math.pi * fc / fs
        h = _h_at(c, w)
        mag_db = 20.0 * math.log10(abs(h))
        assert mag_db == pytest.approx(MINUS_3_0103_DB, abs=1e-9)


def test_coefficients_are_plain_python_floats():
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    c = filt.ideal_coefficients()
    for value in (c.b0, c.b1, c.b2, c.a1, c.a2):
        assert type(value) is float


def test_derived_params_empty():
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    assert filt.derived_params() == {}


def test_kind():
    assert LowPassFilter.kind == "LP"


@pytest.mark.parametrize("bad_fc", [99.9, 0.0, -100.0])
def test_fc_below_minimum_raises(bad_fc):
    with pytest.raises(ValueError):
        LowPassFilter(fs=13333.0, fc=bad_fc)


def test_fc_above_fc_max_raises():
    fs = 13333.0
    with pytest.raises(ValueError):
        LowPassFilter(fs=fs, fc=fc_max(fs) + 1.0)


@pytest.mark.parametrize("bad_fs", [4_999.0, 40_001.0, 0.0, -1000.0])
def test_fs_out_of_range_raises(bad_fs):
    with pytest.raises(ValueError):
        LowPassFilter(fs=bad_fs, fc=1000.0)


def test_q14_coefficients_dispatches_to_backend(fake_backend):
    filt = LowPassFilter(fs=13333.0, fc=3000.0)
    filt.q14_coefficients(fake_backend)
    assert fake_backend.calls == [("design_lp", (3000.0, 13333.0))]
