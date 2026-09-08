"""Sine signal block (docs/CONCEPT.md §11.2)."""

from __future__ import annotations

import numpy as np

from .base import SignalDesign


class SineSignal(SignalDesign):
    kind = "SIN"

    def __init__(self, frequency: float, amplitude: float = 1.0, phase_deg: float = 0.0) -> None:
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.phase_deg = float(phase_deg)
        self.validate()

    def validate(self) -> None:
        if not self.frequency > 0:
            raise ValueError(f"frequency must be > 0 Hz, got {self.frequency}")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        t = np.arange(n_samples, dtype=np.float64) / fs
        return self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + np.radians(self.phase_deg))
