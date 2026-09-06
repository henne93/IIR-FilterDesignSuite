import pytest

from filters.base import Coefficients, FilterDesign, NativeBackend, Q14Coefficients, fc_max


def test_coefficients_as_ba():
    c = Coefficients(b0=1.0, b1=2.0, b2=3.0, a1=4.0, a2=5.0)
    b, a = c.as_ba()
    assert b == [1.0, 2.0, 3.0]
    assert a == [1.0, 4.0, 5.0]


def test_coefficients_frozen():
    c = Coefficients(b0=1.0, b1=2.0, b2=3.0, a1=4.0, a2=5.0)
    with pytest.raises(Exception):
        c.b0 = 99.0


def test_q14_scale_and_to_float():
    assert Q14Coefficients.SCALE == 16384
    q = Q14Coefficients(b0=1677, b1=3354, b2=1677, a1=-22253, a2=9695)
    c = q.to_float()
    assert c.b0 == pytest.approx(0.10235595703125, abs=1e-12)
    assert c.a1 == pytest.approx(-1.35821533203125, abs=1e-12)


@pytest.mark.parametrize("fs", [5_000.0, 8_000.0, 13_333.0, 22_050.0, 40_000.0])
def test_fc_max_reduces_to_0_45_fs_for_supported_fs(fs):
    # CONTRACTS.md §5 note: the 20,000 Hz ceiling never binds within [5_000, 40_000].
    assert fc_max(fs) == pytest.approx(0.45 * fs)
    assert fc_max(fs) < 20_000.0


def test_fc_max_absolute_ceiling_would_bind_above_supported_range():
    assert fc_max(50_000.0) == 20_000.0


def test_filter_design_is_abstract():
    with pytest.raises(TypeError):
        FilterDesign(fs=13333.0)


def test_native_backend_is_runtime_checkable_protocol(fake_backend):
    assert isinstance(fake_backend, NativeBackend)

    class NotABackend:
        pass

    assert not isinstance(NotABackend(), NativeBackend)
