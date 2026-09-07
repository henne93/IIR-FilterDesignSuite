"""2nd-order Peaking EQ (parametric bell boost/cut), per CONTRACTS.md §3.

Derived from the analog peaking-EQ prototype
`H(s) = (s^2 + (A/Q)s + 1) / (s^2 + s/(A*Q) + 1)` (A = 10^(gain_dB/40)),
bilinear-transformed via `s = (1/K)*(1-z^-1)/(1+z^-1)`, `K = tan(pi*fc/fs)` --
the same prewarping used by LP/HP/BP to place the prototype's normalized
critical frequency (Omega=1) exactly at the digital fc, unlike AP's
un-prewarped digital-w0 formula. Pre-warping preserves the critical point
exactly, so `|H(fc)| == gain_dB` dB exactly for any `0 < fc < fs/2`; at DC
and Nyquist the prototype's s=0/s->inf boundary conditions collapse to
`H == 1` (0 dB) exactly, independent of gain/Q.

Q_MIN is 0.8, not All-Pass's 0.25: at low Q combined with high |gain_dB|, the
peaking-EQ coefficients (unlike LP/HP/BP/AP's) are not bounded by ~2.0 in
magnitude -- e.g. Q=0.25 with gain_dB=+15 pushes b0 up to ~3.11, well past
what int16 Q14 (max ~2.0) can represent without real saturation, not just
quantization noise. The true worst case in-domain, at Q_MIN=0.8 and the full
[-15, 15] dB gain range, is |b1|=|a1|~=1.998 (near fc=100 Hz, Q=4.0,
gain_dB=+/-15) -- tight but confirmed non-saturating by
tests/test_native_coefficients.py's domain sweep, same margin class as the
other filter types' worst cases.
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from .base import Coefficients, FilterDesign, NativeBackend, Q14Coefficients, validate_edge_frequency

Q_MIN = 0.8
Q_MAX = 4.0
GAIN_MIN_DB = -15.0
GAIN_MAX_DB = 15.0


class PeakFilter(FilterDesign):
    kind: ClassVar[Literal["PK"]] = "PK"

    fc: float
    Q: float
    gain_db: float

    def validate(self) -> None:
        super().validate()
        validate_edge_frequency("fc", self.fc, self.fs)
        if not (Q_MIN <= self.Q <= Q_MAX):
            raise ValueError(f"Q must be in [{Q_MIN}, {Q_MAX}], got {self.Q}")
        if not (GAIN_MIN_DB <= self.gain_db <= GAIN_MAX_DB):
            raise ValueError(f"gain_db must be in [{GAIN_MIN_DB}, {GAIN_MAX_DB}], got {self.gain_db}")

    def ideal_coefficients(self) -> Coefficients:
        K = math.tan(math.pi * self.fc / self.fs)
        A = 10.0 ** (self.gain_db / 40.0)
        K2 = K * K
        alpha_num = (A / self.Q) * K
        alpha_den = K / (A * self.Q)
        norm = K2 + alpha_den + 1.0
        b0 = (K2 + alpha_num + 1.0) / norm
        b1 = 2.0 * (K2 - 1.0) / norm
        b2 = (K2 - alpha_num + 1.0) / norm
        a1 = b1
        a2 = (K2 - alpha_den + 1.0) / norm
        return Coefficients(b0=b0, b1=b1, b2=b2, a1=a1, a2=a2)

    def q14_coefficients(self, backend: NativeBackend) -> Q14Coefficients:
        return backend.design_pk(self.fc, self.fs, self.Q, self.gain_db)

    def derived_params(self) -> dict[str, float]:
        return {}
