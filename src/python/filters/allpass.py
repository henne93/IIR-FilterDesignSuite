"""2nd-order All-Pass phase shifter, per CONTRACTS.md §3.

Unity magnitude is a structural identity of the mirrored-numerator biquad
(true for any stable a1/a2, independent of Q) — no prewarping is needed,
unlike BP, because the defining properties hold for the un-prewarped
digital w0 directly.
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from .base import Coefficients, FilterDesign, NativeBackend, Q14Coefficients, validate_edge_frequency

Q_MIN = 0.25
Q_MAX = 4.0


class AllPassFilter(FilterDesign):
    kind: ClassVar[Literal["AP"]] = "AP"

    fc: float
    Q: float

    def validate(self) -> None:
        super().validate()
        validate_edge_frequency("fc", self.fc, self.fs)
        if not (Q_MIN <= self.Q <= Q_MAX):
            raise ValueError(f"Q must be in [{Q_MIN}, {Q_MAX}], got {self.Q}")

    def ideal_coefficients(self) -> Coefficients:
        w0 = 2.0 * math.pi * self.fc / self.fs
        alpha = math.sin(w0) / (2.0 * self.Q)
        norm = 1.0 + alpha
        a1 = -2.0 * math.cos(w0) / norm
        a2 = (1.0 - alpha) / norm
        b0 = a2
        b1 = a1
        b2 = 1.0
        return Coefficients(b0=b0, b1=b1, b2=b2, a1=a1, a2=a2)

    def q14_coefficients(self, backend: NativeBackend) -> Q14Coefficients:
        return backend.design_ap(self.fc, self.fs, self.Q)

    def derived_params(self) -> dict[str, float]:
        return {}
