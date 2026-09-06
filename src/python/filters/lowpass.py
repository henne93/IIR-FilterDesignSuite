"""2nd-order Butterworth Low-Pass, per CONTRACTS.md §3."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from .base import Coefficients, FilterDesign, NativeBackend, Q14Coefficients, validate_edge_frequency


class LowPassFilter(FilterDesign):
    kind: ClassVar[Literal["LP"]] = "LP"

    fc: float

    def validate(self) -> None:
        super().validate()
        validate_edge_frequency("fc", self.fc, self.fs)

    def ideal_coefficients(self) -> Coefficients:
        K = math.tan(math.pi * self.fc / self.fs)
        norm = 1.0 + math.sqrt(2.0) * K + K * K
        b0 = b2 = (K * K) / norm
        b1 = 2.0 * b0
        a1 = 2.0 * (K * K - 1.0) / norm
        a2 = (K * K - math.sqrt(2.0) * K + 1.0) / norm
        return Coefficients(b0=b0, b1=b1, b2=b2, a1=a1, a2=a2)

    def q14_coefficients(self, backend: NativeBackend) -> Q14Coefficients:
        return backend.design_lp(self.fc, self.fs)
