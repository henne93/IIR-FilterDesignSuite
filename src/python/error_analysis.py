"""Response- and coefficient-error analysis, per CONTRACTS.md §4, §6, §11, §12.

Phase 3 module: combines the Phase 1 filter models (`filters/*.py`) and the
Phase 2 `NativeBackend` into the two diagnostics CONCEPT.md §6 describes --
selected-filter response validation and the independent coefficient-accuracy
sweep -- plus the series-cascade combined response CONTRACTS.md §12 defines.
No filter design equations are reimplemented here: every comparison goes
through `FilterDesign.ideal_coefficients()` / `.q14_coefficients(backend)`.

Response error is gated to frequency points where the ideal magnitude is
above `RESPONSE_MAGNITUDE_FLOOR_DB` -- see CONTRACTS.md §11 for why deep
in the stopband, Q14 coefficient quantization creates a real noise floor
that makes dB-domain comparison meaningless below that floor, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.signal import freqz

from filters.base import Coefficients, FilterDesign, NativeBackend, fc_max

BODE_GRID_N = 500
BODE_FLOOR_HZ = 10.0

COEFFICIENT_SWEEP_N = 1_000
COEFFICIENT_SWEEP_FLOOR_HZ = 100.0

RESPONSE_MAGNITUDE_FLOOR_DB = -20.0

COEFFICIENT_NAMES: tuple[str, ...] = ("b0", "b1", "b2", "a1", "a2")


def _bode_range(fs: float) -> float:
    """Shared `[BODE_FLOOR_HZ, f_max]` upper bound for `bode_grid`/`linear_response_grid`."""
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    f_max = min(0.7 * fs, fs / 2.0 - 1.0)
    if f_max <= BODE_FLOOR_HZ:
        raise ValueError(f"fs={fs} leaves no valid Bode grid above {BODE_FLOOR_HZ} Hz (f_max={f_max})")
    return f_max


def bode_grid(fs: float, n: int = BODE_GRID_N) -> np.ndarray:
    """Fixed logarithmic Bode/response-validation grid (CONTRACTS.md §6.1/§6.2).

    `n` log-spaced points from `BODE_FLOOR_HZ` to `min(0.7*fs, fs/2 - 1)`. This
    is the same grid used for both the Bode display and the selected-filter
    response-error comparison -- CONTRACTS.md §6 is explicit that these are
    not two separate definitions.
    """
    f_max = _bode_range(fs)
    return np.logspace(np.log10(BODE_FLOOR_HZ), np.log10(f_max), n)


def linear_response_grid(fs: float, n: int = BODE_GRID_N) -> np.ndarray:
    """Linear-frequency counterpart of `bode_grid()`, for the linear gain plot.

    Same `[BODE_FLOOR_HZ, min(0.7*fs, fs/2 - 1)]` range as `bode_grid()` --
    only the spacing differs (linear, not logarithmic).
    """
    f_max = _bode_range(fs)
    return np.linspace(BODE_FLOOR_HZ, f_max, n)


@dataclass(frozen=True)
class ResponseError:
    """Amplitude-error result for one ideal-vs-Q14 response comparison.

    `freq_hz`/`error_db` only cover points that passed the
    `RESPONSE_MAGNITUDE_FLOOR_DB` gate (CONTRACTS.md §11) -- they are shorter
    than the input grid whenever some of it falls in the deep stopband.
    """

    freq_hz: np.ndarray
    error_db: np.ndarray
    max_db: float
    rms_db: float  # informational only -- no pass/fail gate, CONTRACTS.md §11


@dataclass(frozen=True)
class CoefficientSweepError:
    """Diagnostic coefficient-accuracy sweep result (CONTRACTS.md §6.3/§11).

    Diagnostic only -- CONTRACTS.md §11 is explicit that the coefficient
    sweep has no pass/fail threshold.
    """

    max_abs: float
    rms_abs: float
    worst_frequency_hz: float
    per_coefficient_max: dict[str, float]


def _require_backend(backend: NativeBackend | None) -> None:
    if backend is None:
        raise ValueError("a NativeBackend is required for Q14 comparisons (none was provided)")
    missing = [
        m for m in ("design_lp", "design_hp", "design_bp", "design_ap", "design_pk") if not hasattr(backend, m)
    ]
    if missing:
        raise ValueError(f"backend is missing required NativeBackend method(s): {', '.join(missing)}")


def _amplitude_db(coeffs: Coefficients, fs: float, freq_hz: np.ndarray) -> np.ndarray:
    w = 2.0 * np.pi * freq_hz / fs
    b, a = coeffs.as_ba()
    _, h = freqz(b, a, worN=w)
    return 20.0 * np.log10(np.abs(h))


def _gated_error(freq_hz: np.ndarray, mag_ideal_db: np.ndarray, mag_q14_db: np.ndarray) -> ResponseError:
    err = np.abs(mag_ideal_db - mag_q14_db)
    mask = mag_ideal_db > RESPONSE_MAGNITUDE_FLOOR_DB
    gated_freq = np.asarray(freq_hz)[mask]
    gated_err = err[mask]
    if gated_err.size == 0:
        raise ValueError(
            "no frequency-grid points had ideal magnitude above the "
            f"{RESPONSE_MAGNITUDE_FLOOR_DB} dB response-error floor"
        )
    return ResponseError(
        freq_hz=gated_freq,
        error_db=gated_err,
        max_db=float(gated_err.max()),
        rms_db=float(np.sqrt(np.mean(gated_err**2))),
    )


def response_error(
    filt: FilterDesign, backend: NativeBackend, freq_hz: np.ndarray | None = None
) -> ResponseError:
    """Selected-filter ideal-vs-Q14 response comparison (CONCEPT.md §6, CONTRACTS.md §4)."""
    _require_backend(backend)
    if freq_hz is None:
        freq_hz = bode_grid(filt.fs)
    ideal = filt.ideal_coefficients()
    q14_float = filt.q14_coefficients(backend).to_float()
    mag_i = _amplitude_db(ideal, filt.fs, freq_hz)
    mag_q = _amplitude_db(q14_float, filt.fs, freq_hz)
    return _gated_error(freq_hz, mag_i, mag_q)


def combined_response_error(
    chain: Sequence[FilterDesign], backend: NativeBackend, freq_hz: np.ndarray | None = None
) -> ResponseError:
    """Series-cascade ideal-vs-Q14 response comparison (CONTRACTS.md §12).

    Cascade magnitudes add in dB (`H_combined = Pi H_i`), so the combined
    ideal/Q14 curves are the per-block dB sums, then gated/compared exactly
    like `response_error`. Raises on an empty chain rather than returning a
    degenerate flat 0 dB line (CONTRACTS.md §12: "Combined tab shows an
    explicit empty-state message").
    """
    _require_backend(backend)
    if not chain:
        raise ValueError("cannot compute a combined response for an empty filter chain")
    fs_values = {f.fs for f in chain}
    if len(fs_values) > 1:
        raise ValueError(f"all filters in a chain must share the same fs, got {sorted(fs_values)}")
    fs = chain[0].fs
    if freq_hz is None:
        freq_hz = bode_grid(fs)
    freq_hz = np.asarray(freq_hz, dtype=np.float64)

    mag_i_total = np.zeros_like(freq_hz)
    mag_q_total = np.zeros_like(freq_hz)
    for filt in chain:
        ideal = filt.ideal_coefficients()
        q14_float = filt.q14_coefficients(backend).to_float()
        mag_i_total += _amplitude_db(ideal, fs, freq_hz)
        mag_q_total += _amplitude_db(q14_float, fs, freq_hz)

    return _gated_error(freq_hz, mag_i_total, mag_q_total)


def coefficient_sweep(
    fs: float,
    design_at: Callable[[float], FilterDesign],
    backend: NativeBackend,
    n: int = COEFFICIENT_SWEEP_N,
) -> CoefficientSweepError:
    """Independent coefficient-accuracy sweep (CONCEPT.md §6, CONTRACTS.md §6.3/§11).

    Sweeps `n` linearly-spaced points from `COEFFICIENT_SWEEP_FLOOR_HZ` to
    `fc_max(fs)`. `design_at(x)` must build a `FilterDesign` at sweep value
    `x` and this `fs`, holding every other parameter fixed -- e.g. for LP/HP/AP
    `x` is `fc` directly; for BP, `x` is the swept quantity (typically center
    frequency) with bandwidth/Q held fixed per CONTRACTS.md §6.3, and the
    caller's factory is responsible for keeping `f_low`/`f_high` valid across
    the full swept range.
    """
    _require_backend(backend)
    hi = fc_max(fs)
    if not (COEFFICIENT_SWEEP_FLOOR_HZ < hi):
        raise ValueError(f"fs={fs} yields an empty coefficient sweep range (fc_max={hi})")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    grid = np.linspace(COEFFICIENT_SWEEP_FLOOR_HZ, hi, n)

    max_abs = 0.0
    worst_freq = float(grid[0])
    per_coefficient_max = {name: 0.0 for name in COEFFICIENT_NAMES}
    squared_sum = 0.0
    count = 0

    for x in grid:
        filt = design_at(float(x))
        ideal = filt.ideal_coefficients()
        q14_float = filt.q14_coefficients(backend).to_float()
        point_max = 0.0
        for name in COEFFICIENT_NAMES:
            err = abs(getattr(ideal, name) - getattr(q14_float, name))
            squared_sum += err * err
            count += 1
            if err > per_coefficient_max[name]:
                per_coefficient_max[name] = err
            if err > point_max:
                point_max = err
        if point_max > max_abs:
            max_abs = point_max
            worst_freq = float(x)

    return CoefficientSweepError(
        max_abs=max_abs,
        rms_abs=float(np.sqrt(squared_sum / count)),
        worst_frequency_hz=worst_freq,
        per_coefficient_max=per_coefficient_max,
    )
