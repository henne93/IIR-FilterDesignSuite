"""Time-domain signal generation and filtering (docs/CONCEPT.md §11.3-§11.5).

Bridges `signals.SignalChain` (source generation) and `filters.FilterChain`
(the ideal/Q14 cascade) into the three plotted curves (source, ideal-
filtered, Q14-filtered) the Time-Domain Inspector renders. Kept Qt-
independent and side-effect-free, mirroring error_analysis.py's role for
the Bode/gain plots -- callers (the UI) generate the source signal from a
`SignalChain` themselves and pass the resulting array in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import lfilter

from filters.base import FilterDesign, NativeBackend

INT16_MIN = -32768
INT16_MAX = 32767


@dataclass(frozen=True)
class TimeDomainResult:
    time_ms: np.ndarray
    source: np.ndarray
    ideal: np.ndarray
    q14: np.ndarray | None  # None iff no NativeBackend was available


def n_samples_for(duration_ms: float, fs: float) -> int:
    """Sample count for a plotted window (§11.3): `round(duration_ms / 1000 * fs)`, at least 1."""
    return max(1, round(duration_ms / 1000.0 * fs))


def to_q14_samples(normalized: np.ndarray, full_scale: int) -> np.ndarray:
    """Maps normalized float samples to int16 counts (§11.4): round to nearest,
    then clip to the int16 range. Clipping is intentional here, not an error --
    letting a signal saturate is exactly the Q14 behavior the full-scale
    reference control is meant to let the user explore/demonstrate.
    """
    scaled = np.round(np.asarray(normalized, dtype=np.float64) * full_scale)
    return np.clip(scaled, INT16_MIN, INT16_MAX).astype(np.int64)


def from_q14_samples(samples: Sequence[int], full_scale: int) -> np.ndarray:
    """Inverse of `to_q14_samples()`: maps int16 counts back to the normalized float axis."""
    return np.asarray(samples, dtype=np.float64) / full_scale


def filter_ideal(source: np.ndarray, filters: Sequence[FilterDesign]) -> np.ndarray:
    """Cascades each filter's `ideal_coefficients()` in series over `source` (§11.5)."""
    signal = np.asarray(source, dtype=np.float64)
    for filt in filters:
        b, a = filt.ideal_coefficients().as_ba()
        signal = lfilter(b, a, signal)
    return signal


def filter_q14(source_int16: Sequence[int], filters: Sequence[FilterDesign], backend: NativeBackend) -> list[int]:
    """Cascades the Q14 native path in series: one block's int16 output feeds the
    next block's input (§11.5), matching how the cascade runs on the target firmware.
    """
    samples = [int(s) for s in source_int16]
    for filt in filters:
        coeffs = filt.q14_coefficients(backend)
        samples = backend.process_samples(coeffs, samples)
    return samples


def compute(
    source: np.ndarray,
    filters: Sequence[FilterDesign],
    fs: float,
    full_scale: int,
    backend: NativeBackend | None,
) -> TimeDomainResult:
    """Computes the three plotted curves for an already-generated `source` signal.

    `filters` is the ordered cascade to apply -- pass `FilterChain.valid_filters`
    for the "Combined" view or `[block.filter]` for a single filter block's tab
    (§11.6). An empty `filters` means "no filtering": `ideal`/`q14` are the
    source signal itself (float-passthrough / Q14-quantized-only respectively),
    which is a normal, valid state, not an error.
    """
    source = np.asarray(source, dtype=np.float64)
    n = len(source)
    time_ms = np.arange(n, dtype=np.float64) / fs * 1000.0
    ideal = filter_ideal(source, filters)

    q14 = None
    if backend is not None:
        source_int16 = to_q14_samples(source, full_scale)
        q14_int = filter_q14(source_int16, filters, backend)
        q14 = from_q14_samples(q14_int, full_scale)

    return TimeDomainResult(time_ms=time_ms, source=source, ideal=ideal, q14=q14)
