"""Signal-block design interfaces for the Time-Domain source generator (docs/CONCEPT.md §11.2).

Mirrors filters/base.py's FilterDesign split: each concrete signal-block
kind (Sine/DC/Noise/CSV) is a small class implementing `generate()`, kept
separate from `SignalChain` (signals/chain.py) exactly like FilterDesign is
kept separate from FilterChain -- SignalChain owns ids/ordering/summation,
signal-block classes own their own per-block math and parameter validation.

Unlike a FilterDesign, a signal block has no `fs` of its own: fs is a
chain-wide setting (the same fs the filter chain uses, §11.3) supplied to
`generate()` at call time rather than fixed at construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np

SignalBlockKind = Literal["SIN", "DC", "NOISE", "CSV"]


class SignalDesign(ABC):
    kind: ClassVar[SignalBlockKind]

    @abstractmethod
    def validate(self) -> None:
        """Raises ValueError with a human-readable message. Never clamps silently."""

    @abstractmethod
    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """Returns this block's own raw signal (`factor` is applied by the caller, §11.2)."""
