"""True 2nd-order Butterworth Band-Pass, dual-edge-prewarped, per CONTRACTS.md §3.

Revised from CONCEPT.md's single-K/linear-bandwidth pseudocode, which only
prewarps the center frequency and misses the exact -3.0103 dB edges for
non-narrow bands (see CONTRACTS.md §3 and "Summary of deviations" item 1).
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from .base import Coefficients, FilterDesign, NativeBackend, Q14Coefficients, validate_edge_frequency


class BandPassFilter(FilterDesign):
    kind: ClassVar[Literal["BP"]] = "BP"

    f_low: float
    f_high: float

    def validate(self) -> None:
        super().validate()
        validate_edge_frequency("f_low", self.f_low, self.fs)
        validate_edge_frequency("f_high", self.f_high, self.fs)
        if not (self.f_low < self.f_high):
            raise ValueError(f"f_low must be strictly less than f_high, got f_low={self.f_low}, f_high={self.f_high}")

    def ideal_coefficients(self) -> Coefficients:
        K_low = math.tan(math.pi * self.f_low / self.fs)
        K_high = math.tan(math.pi * self.f_high / self.fs)
        wc2 = K_low * K_high
        BW = K_high - K_low
        norm = 1.0 + BW + wc2
        b0 = BW / norm
        b1 = 0.0
        b2 = -BW / norm
        a1 = 2.0 * (wc2 - 1.0) / norm
        a2 = (1.0 - BW + wc2) / norm
        return Coefficients(b0=b0, b1=b1, b2=b2, a1=a1, a2=a2)

    def q14_coefficients(self, backend: NativeBackend) -> Q14Coefficients:
        return backend.design_bp(self.f_low, self.f_high, self.fs)

    def derived_params(self) -> dict[str, float]:
        fc = math.sqrt(self.f_low * self.f_high)
        q = fc / (self.f_high - self.f_low)
        return {"fc": fc, "Q": q}
