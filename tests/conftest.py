import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "python"))

from filters.base import Q14Coefficients  # noqa: E402

# fs values spanning the supported [5_000, 40_000] Hz domain (CONTRACTS.md §5).
FS_SWEEP = [5_000.0, 8_000.0, 13_333.0, 22_050.0, 40_000.0]


class FakeNativeBackend:
    """Duck-typed stand-in for the Phase-2 ctypes NativeBackend.

    Records call arguments instead of doing real Q14 quantization, so Phase-1
    filter classes can be exercised through q14_coefficients() without a
    compiled library (CONTRACTS.md §1: "keeps the filter classes testable
    without a compiled library present").
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def design_lp(self, fc: float, fs: float) -> Q14Coefficients:
        self.calls.append(("design_lp", (fc, fs)))
        return Q14Coefficients(0, 0, 0, 0, 0)

    def design_hp(self, fc: float, fs: float) -> Q14Coefficients:
        self.calls.append(("design_hp", (fc, fs)))
        return Q14Coefficients(0, 0, 0, 0, 0)

    def design_bp(self, f_low: float, f_high: float, fs: float) -> Q14Coefficients:
        self.calls.append(("design_bp", (f_low, f_high, fs)))
        return Q14Coefficients(0, 0, 0, 0, 0)

    def design_ap(self, fc: float, fs: float, q: float) -> Q14Coefficients:
        self.calls.append(("design_ap", (fc, fs, q)))
        return Q14Coefficients(0, 0, 0, 0, 0)

    def design_pk(self, fc: float, fs: float, q: float, gain_db: float) -> Q14Coefficients:
        self.calls.append(("design_pk", (fc, fs, q, gain_db)))
        return Q14Coefficients(0, 0, 0, 0, 0)


@pytest.fixture
def fake_backend() -> FakeNativeBackend:
    return FakeNativeBackend()


@pytest.fixture(params=FS_SWEEP)
def fs(request) -> float:
    return request.param


def edge_fcs(fs: float, n: int = 5) -> np.ndarray:
    """A handful of fc/edge values spanning [100, fc_max(fs)] for a given fs."""
    from filters.base import fc_max

    return np.linspace(100.0, fc_max(fs), n)


@pytest.fixture(scope="session")
def native_backend():
    """Compiles src/c/*.c once per test session (CONCEPT.md's conftest.py role).

    Phase 2 (CONTRACTS.md §2/§8/§9): a real ctypes-backed NativeBackend,
    reused across every native test in the session rather than recompiled
    per test -- still a single process-local compile, matching the "cached
    until next launch" (== next process) semantics of §9.
    """
    from c_codegen import NativeBackend

    backend = NativeBackend()
    yield backend
    backend.close()
