"""Additive white Gaussian noise signal block (docs/CONCEPT.md §11.2)."""

from __future__ import annotations

import numpy as np

from .base import SignalDesign


class NoiseSignal(SignalDesign):
    """AWGN: samples drawn from N(0, amplitude^2).

    `seed` is fixed at construction (chosen by `SignalChain.add_block()`
    when the caller doesn't supply one) rather than reseeded on every
    `generate()` call, so the plotted curve stays stable across the app's
    live re-renders (an unrelated field's `editingFinished`, an fs echo,
    duration edits, ...) and only changes if this block's own params are
    edited. This is a deliberate choice -- confirmed with the user -- over
    freshly-drawn noise on every redraw, which would make the plot jitter
    on every unrelated UI update. `SignalChain.reseed()` (wired to the
    Noise tile's "Reseed" button, `ui/widgets/signal_block.py`) is the one
    explicit, user-triggered way to change it.
    """

    kind = "NOISE"

    def __init__(self, amplitude: float = 0.05, seed: int = 0) -> None:
        self.amplitude = float(amplitude)
        self.seed = int(seed)
        self.validate()

    def validate(self) -> None:
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be >= 0, got {self.amplitude}")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, self.amplitude, size=n_samples)
