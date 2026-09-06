import cmath
import math

import numpy as np
import pytest

from filters.allpass import AllPassFilter
from filters.base import fc_max


def _h_at(coeffs, w):
    z_inv = cmath.exp(-1j * w)
    num = coeffs.b0 + coeffs.b1 * z_inv + coeffs.b2 * z_inv**2
    den = 1.0 + coeffs.a1 * z_inv + coeffs.a2 * z_inv**2
    return num / den


def test_independent_complex_arithmetic_cross_check(fs):
    fc = 0.4 * fc_max(fs) + 0.3 * 100.0
    for q in (0.25, 1.0 / (2 ** 0.5), 4.0):
        c = AllPassFilter(fs=fs, fc=fc, Q=q).ideal_coefficients()
        for w in (0.1, 1.0, 2.0 * math.pi * fc / fs, 3.0):
            h = _h_at(c, w)
            assert abs(h) == pytest.approx(1.0, abs=1e-9)
        w0 = 2.0 * math.pi * fc / fs
        phase = math.degrees(cmath.phase(_h_at(c, w0))) % 360.0
        assert phase == pytest.approx(180.0, abs=1e-6)


@pytest.mark.parametrize("q", [0.25, 0.5, 1.0 / (2 ** 0.5), 2.0, 4.0])
def test_unity_magnitude_everywhere(fs, q):
    fc = 0.4 * fc_max(fs) + 0.3 * 100.0
    filt = AllPassFilter(fs=fs, fc=fc, Q=q)
    freqs = np.logspace(np.log10(10.0), np.log10(fs / 2.0 - 1.0), 200)
    response = filt.ideal_response(freqs)
    assert response.magnitude_db == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("q", [0.25, 1.0 / (2 ** 0.5), 4.0])
def test_phase_is_minus_180_at_fc(fs, q):
    fc = 0.4 * fc_max(fs) + 0.3 * 100.0
    filt = AllPassFilter(fs=fs, fc=fc, Q=q)
    response = filt.ideal_response(np.array([fc]))
    phase = response.phase_deg[0] % 360.0
    assert phase == pytest.approx(180.0, abs=1e-6)


def test_default_butterworth_q_is_valid():
    q = 1.0 / (2 ** 0.5)
    filt = AllPassFilter(fs=13333.0, fc=3000.0, Q=q)
    assert filt.Q == pytest.approx(q)


def test_derived_params_empty():
    filt = AllPassFilter(fs=13333.0, fc=3000.0, Q=0.707)
    assert filt.derived_params() == {}


def test_kind():
    assert AllPassFilter.kind == "AP"


@pytest.mark.parametrize("bad_q", [0.24, 4.01, 0.0, -1.0])
def test_q_out_of_range_raises(bad_q):
    with pytest.raises(ValueError):
        AllPassFilter(fs=13333.0, fc=3000.0, Q=bad_q)


def test_fc_out_of_range_raises():
    fs = 13333.0
    with pytest.raises(ValueError):
        AllPassFilter(fs=fs, fc=fc_max(fs) + 1.0, Q=0.707)
    with pytest.raises(ValueError):
        AllPassFilter(fs=fs, fc=99.0, Q=0.707)


def test_q14_coefficients_dispatches_to_backend(fake_backend):
    filt = AllPassFilter(fs=13333.0, fc=3000.0, Q=0.707)
    filt.q14_coefficients(fake_backend)
    assert fake_backend.calls == [("design_ap", (3000.0, 13333.0, 0.707))]
