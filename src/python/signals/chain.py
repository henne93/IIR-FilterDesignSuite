"""Qt-independent signal-chain model for the Time-Domain source generator (docs/CONCEPT.md §11.2).

Analogous to filters/chain.py's `FilterChain`, but blocks *sum* rather than
cascade (order doesn't matter, §11.2/§11.7) and there is no chain-wide `fs`
here: fs is supplied by the caller at generation time (the same fs the
`FilterChain` already owns, per §11.3 -- there is exactly one fs in the
app). For the same reason there is no block ordering/move operation either:
a flat, unordered collection feeding a sum has no "position" that matters.

Per-block parameter invalidity never raises (mirrors FilterChain's
CONTRACTS.md §5 philosophy): `add_block()`/`update_params()` catch the
underlying `SignalDesign`'s `ValueError` and store it on the
`SignalChainBlock` (`.error`) instead of propagating it. An invalid block is
excluded from `source_signal()` exactly like the invalid-block exclusion in
`FilterChain.valid_filters`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .base import SignalBlockKind, SignalDesign
from .csv_import import CsvSignal
from .dc import DcSignal
from .noise import NoiseSignal
from .sine import SineSignal

_KIND_TO_CLASS: dict[SignalBlockKind, type[SignalDesign]] = {
    "SIN": SineSignal,
    "DC": DcSignal,
    "NOISE": NoiseSignal,
    "CSV": CsvSignal,
}

# Defaults for a freshly-added block (§11.4: normalized floats, -1.0 to +1.0
# full scale). NOISE's "seed" is deliberately absent here -- add_block()
# fills in a fresh random seed unless the caller supplies one explicitly.
DEFAULT_SIGNAL_PARAMS: dict[SignalBlockKind, dict[str, float | str]] = {
    "SIN": {"frequency": 1_000.0, "amplitude": 0.5, "phase_deg": 0.0},
    "DC": {"value": 0.0},
    "NOISE": {"amplitude": 0.05},
    "CSV": {"file_path": ""},
}

DEFAULT_FACTOR = 1.0


def _build(kind: SignalBlockKind, params: Mapping[str, float | str]) -> SignalDesign:
    return _KIND_TO_CLASS[kind](**params)


@dataclass(frozen=True)
class SignalChainBlock:
    """One signal-chain entry: stable id, kind, current params/factor, and validity state."""

    id: str
    kind: SignalBlockKind
    params: Mapping[str, float | str]
    factor: float
    design: SignalDesign | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.design is not None


def _make_block(
    block_id: str, kind: SignalBlockKind, params: Mapping[str, float | str], factor: float
) -> SignalChainBlock:
    frozen_params = MappingProxyType(dict(params))
    try:
        design = _build(kind, frozen_params)
        return SignalChainBlock(
            id=block_id, kind=kind, params=frozen_params, factor=factor, design=design, error=None
        )
    except ValueError as exc:
        return SignalChainBlock(
            id=block_id, kind=kind, params=frozen_params, factor=factor, design=None, error=str(exc)
        )


class SignalChain:
    """The additive signal-generator model backing the Time-Domain view's canvas/inspector."""

    def __init__(self) -> None:
        self._blocks: list[SignalChainBlock] = []
        self._next_id = itertools.count(1)

    @property
    def blocks(self) -> list[SignalChainBlock]:
        """Ordered snapshot of the chain's blocks. Mutate via chain methods, not this list."""
        return list(self._blocks)

    @property
    def valid_blocks(self) -> list[SignalChainBlock]:
        return [b for b in self._blocks if b.design is not None]

    def get_block(self, block_id: str) -> SignalChainBlock:
        for b in self._blocks:
            if b.id == block_id:
                return b
        raise KeyError(f"no signal block with id {block_id!r}")

    def _index_of(self, block_id: str) -> int:
        for i, b in enumerate(self._blocks):
            if b.id == block_id:
                return i
        raise KeyError(f"no signal block with id {block_id!r}")

    # -- mutation -----------------------------------------------------------

    def add_block(self, kind: SignalBlockKind, factor: float = DEFAULT_FACTOR, **params: float | str) -> str:
        """Appends a new block, returning its stable id.

        `params` override `DEFAULT_SIGNAL_PARAMS[kind]` per-key; omitted
        params use the documented default. Invalid params do not raise --
        the block is still added, marked invalid, with `.error` set (see
        module docstring).
        """
        if kind not in _KIND_TO_CLASS:
            raise ValueError(f"unknown signal block kind {kind!r}, expected one of {sorted(_KIND_TO_CLASS)}")
        resolved: dict[str, float | str] = dict(DEFAULT_SIGNAL_PARAMS[kind])
        resolved.update(params)
        if kind == "NOISE" and "seed" not in resolved:
            resolved["seed"] = int(np.random.default_rng().integers(0, 2**31 - 1))
        block_id = f"sig{next(self._next_id)}"
        self._blocks.append(_make_block(block_id, kind, resolved, float(factor)))
        return block_id

    def remove_block(self, block_id: str) -> None:
        idx = self._index_of(block_id)
        del self._blocks[idx]

    def update_params(self, block_id: str, **params: float | str) -> None:
        """Merges `params` into the block's current params and re-validates.

        Never raises for an invalid result (see module docstring) -- check
        `chain.get_block(block_id).is_valid` / `.error` afterward. `factor`
        is a separate field (`set_factor()`), not a `params` key.
        """
        idx = self._index_of(block_id)
        old = self._blocks[idx]
        merged = dict(old.params)
        merged.update(params)
        self._blocks[idx] = _make_block(old.id, old.kind, merged, old.factor)

    def set_factor(self, block_id: str, factor: float) -> None:
        """Updates only the uniform gain knob (§11.2) -- never re-validates `design`,
        since a block's own params' validity never depends on `factor`."""
        idx = self._index_of(block_id)
        self._blocks[idx] = replace(self._blocks[idx], factor=float(factor))

    def clear(self) -> None:
        self._blocks = []

    # -- validity ------------------------------------------------------------

    @property
    def has_invalid_blocks(self) -> bool:
        return any(not b.is_valid for b in self._blocks)

    def validation_messages(self) -> dict[str, str]:
        """`{block_id: error_message}` for every currently-invalid block."""
        return {b.id: b.error for b in self._blocks if b.error is not None}

    # -- source signal ------------------------------------------------------------

    def source_signal(self, n_samples: int, fs: float) -> np.ndarray:
        """Sums every valid block's `factor * design.generate(...)` (§11.2).

        Invalid blocks are excluded exactly like `FilterChain.valid_filters`
        excludes invalid filter blocks.
        """
        if not self._blocks:
            raise ValueError("cannot generate a source signal: signal chain is empty")
        valid = self.valid_blocks
        if not valid:
            raise ValueError(f"cannot generate a source signal: all {len(self._blocks)} block(s) are invalid")
        total = np.zeros(n_samples, dtype=np.float64)
        for block in valid:
            total += block.factor * block.design.generate(n_samples, fs)
        return total
