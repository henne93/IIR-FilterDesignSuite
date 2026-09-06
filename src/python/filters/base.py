"""Filter-model interfaces and shared math for the IIR Filter Design Suite.

Implements the contracts in docs/CONTRACTS.md §1 (interfaces/dataclasses),
§2 (coefficient ordering/sign convention), §4 ("ideal" = closed-form §3
equations in float64, scipy used only as the freqz evaluator), §5
(parameter ranges / fc_max), and §6 (freqz called with an explicit angular
frequency array, never scipy's own worN=N grid).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from scipy.signal import freqz

FS_MIN_HZ = 5_000.0
FS_MAX_HZ = 40_000.0
FC_ABSOLUTE_CEILING_HZ = 20_000.0
FC_MIN_HZ = 100.0


def fc_max(fs: float) -> float:
    """Effective digital cutoff/edge maximum for a given fs (CONTRACTS.md §5)."""
    return min(FC_ABSOLUTE_CEILING_HZ, 0.45 * fs)


@dataclass(frozen=True)
class Coefficients:
    """H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2). a0 is always 1."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def as_ba(self) -> tuple[list[float], list[float]]:
        """Returns (b, a) exactly as scipy.signal.freqz expects."""
        return [self.b0, self.b1, self.b2], [1.0, self.a1, self.a2]


@dataclass(frozen=True)
class Q14Coefficients:
    """Q14 fixed-point, scale = 16384. Storage width is int32_t (see CONTRACTS.md §7)."""

    b0: int
    b1: int
    b2: int
    a1: int
    a2: int
    SCALE: ClassVar[int] = 16384

    def to_float(self) -> Coefficients:
        s = self.SCALE
        return Coefficients(self.b0 / s, self.b1 / s, self.b2 / s, self.a1 / s, self.a2 / s)


@dataclass(frozen=True)
class FrequencyResponse:
    freq_hz: np.ndarray  # (N,) strictly increasing, float64
    magnitude_db: np.ndarray  # (N,) float64
    phase_deg: np.ndarray  # (N,) float64, unwrapped


@runtime_checkable
class NativeBackend(Protocol):
    """Handle to the compiled Q14 design functions (CONTRACTS.md §8).

    Injected into FilterDesign.q14_coefficients()/q14_response() rather than
    used as a module-level singleton, so filter classes stay testable without
    a compiled library present. The real implementation (Phase 2, ctypes-backed)
    lives in c_codegen.py; this Protocol only fixes the shape it must have.
    """

    def design_lp(self, fc: float, fs: float) -> Q14Coefficients: ...
    def design_hp(self, fc: float, fs: float) -> Q14Coefficients: ...
    def design_bp(self, f_low: float, f_high: float, fs: float) -> Q14Coefficients: ...
    def design_ap(self, fc: float, fs: float, q: float) -> Q14Coefficients: ...


def _response_from_coefficients(coeffs: Coefficients, freq_hz: np.ndarray, fs: float) -> FrequencyResponse:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    b, a = coeffs.as_ba()
    w = 2.0 * np.pi * freq_hz / fs
    _, h = freqz(b, a, worN=w)
    magnitude_db = 20.0 * np.log10(np.abs(h))
    phase_deg = np.degrees(np.unwrap(np.angle(h)))
    return FrequencyResponse(freq_hz=freq_hz, magnitude_db=magnitude_db, phase_deg=phase_deg)


class FilterDesign(ABC):
    kind: ClassVar[Literal["LP", "HP", "BP", "AP"]]

    def __init__(self, fs: float, **params: float) -> None:
        self.fs: float = float(fs)
        for name, value in params.items():
            setattr(self, name, float(value))
        self.validate()

    @abstractmethod
    def ideal_coefficients(self) -> Coefficients: ...

    @abstractmethod
    def q14_coefficients(self, backend: NativeBackend) -> Q14Coefficients: ...

    def ideal_response(self, freq_hz: np.ndarray) -> FrequencyResponse:
        return _response_from_coefficients(self.ideal_coefficients(), freq_hz, self.fs)

    def q14_response(self, freq_hz: np.ndarray, backend: NativeBackend) -> FrequencyResponse:
        coeffs = self.q14_coefficients(backend).to_float()
        return _response_from_coefficients(coeffs, freq_hz, self.fs)

    def validate(self) -> None:
        """Raises ValueError with a human-readable message. Never clamps silently."""
        if not (FS_MIN_HZ <= self.fs <= FS_MAX_HZ):
            raise ValueError(f"fs must be in [{FS_MIN_HZ}, {FS_MAX_HZ}] Hz, got {self.fs}")

    def derived_params(self) -> dict[str, float]:
        """e.g. BP -> {'fc': ..., 'Q': ...}. Empty dict for LP/HP. AP -> {} (Q is a primary param)."""
        return {}


def validate_edge_frequency(name: str, value: float, fs: float) -> None:
    """Shared [100, fc_max(fs)] range check for fc / f_low / f_high (CONTRACTS.md §5)."""
    hi = fc_max(fs)
    if not (FC_MIN_HZ <= value <= hi):
        raise ValueError(f"{name} must be in [{FC_MIN_HZ}, {hi}] Hz for fs={fs} Hz, got {value}")
