"""DC (constant offset) signal block (docs/CONCEPT.md §11.2)."""

from __future__ import annotations

import numpy as np

from .base import SignalDesign


class DcSignal(SignalDesign):
    kind = "DC"

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)
        self.validate()

    def validate(self) -> None:
        pass

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        return np.full(n_samples, self.value, dtype=np.float64)
