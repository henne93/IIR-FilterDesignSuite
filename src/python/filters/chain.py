"""Qt-independent filter chain / application model (CONTRACTS.md §12, §13).

Phase 4: the FilterChain model owns the chain-wide `fs` and the ordered list
of filter blocks, decoupled from Qt so it is unit-testable without a running
UI (CONTRACTS.md §12's "Proposed addition"). `canvas.py` (Phase 5) is
expected to hold only indices/ids into this model and re-render from it --
single source of truth, per CONTRACTS.md §12's ownership rule.

Design decisions beyond CONTRACTS.md §12's draft pseudocode (flagged there as
a "Proposed addition", not a locked interface -- these are deliberate
phase-4 choices, not deviations from a contract):

- **Blocks carry stable string ids** (`"blk1"`, `"blk2"`, ...), assigned once
  in `add_block()` and never reused or renumbered by reorder/delete. The
  draft pseudocode's `add_block() -> int` (a position index) would not
  survive a reorder or the deletion of an earlier block -- exactly what this
  phase's "stable block IDs independent of list position" requirement rules
  out. Every block-targeting method (`remove_block`, `move_block`,
  `update_params`, `select`) takes an id, never an index. Display/export
  numbering (`FILT<n>`, CONTRACTS.md §10) is a separate, position-derived
  concern for the caller (`enumerate(chain.blocks)`), not something this
  model tracks.
- **Per-block parameter invalidity never raises.** `add_block()`,
  `update_params()`, and fs-triggered revalidation all catch the underlying
  `FilterDesign`'s `ValueError` and store it on the `ChainBlock` (`.error`)
  instead of propagating it -- CONTRACTS.md §5's "re-validates every block;
  ... gets an error badge" cannot be satisfied by letting one bad block's
  constructor exception abort revalidation of the rest of the chain. `fs`
  itself is the one exception: the `FilterChain.fs` setter *does* raise
  `ValueError` immediately for a value outside `[5000, 40000]` Hz, matching
  CONTRACTS.md §5's literal "raised from the constructor/setter" language --
  there is exactly one `fs`, so there is no "which block" to badge instead.
  Parameters are never silently clamped in either path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from .allpass import AllPassFilter
from .bandpass import BandPassFilter
from .base import FS_MAX_HZ, FS_MIN_HZ, FilterDesign, FrequencyResponse, NativeBackend
from .highpass import HighPassFilter
from .lowpass import LowPassFilter
from .peak import PeakFilter

BlockKind = Literal["LP", "HP", "BP", "AP", "PK"]

_KIND_TO_CLASS: dict[BlockKind, type[FilterDesign]] = {
    "LP": LowPassFilter,
    "HP": HighPassFilter,
    "BP": BandPassFilter,
    "AP": AllPassFilter,
    "PK": PeakFilter,
}

# Defaults for a freshly-added block. Chosen well inside every parameter's
# valid range (CONTRACTS.md §5) for *every* supported fs: the tightest
# fc_max(fs) across the supported [5_000, 40_000] Hz domain is
# fc_max(5_000) = 2_250 Hz, so a 1_000/2_000 Hz default is always valid
# regardless of the chain's current fs. AP's Q = 1/sqrt(2) is the
# Butterworth-flat default, comfortably inside [0.25, 4.0]. PK's own Q range
# is [0.8, 4.0] (narrower than AP's -- see filters/peak.py's module
# docstring for why low Q + high gain can exceed int16 Q14 headroom), so its
# default Q=1.0 is used instead of 1/sqrt(2); default gain is +6 dB (inside
# [-15, 15]).
DEFAULT_PARAMS: dict[BlockKind, dict[str, float]] = {
    "LP": {"fc": 1_000.0},
    "HP": {"fc": 1_000.0},
    "BP": {"f_low": 1_000.0, "f_high": 2_000.0},
    "AP": {"fc": 1_000.0, "Q": 1.0 / 2.0**0.5},
    "PK": {"fc": 1_000.0, "Q": 1.0, "gain_db": 6.0},
}


def _build(kind: BlockKind, fs: float, params: Mapping[str, float]) -> FilterDesign:
    return _KIND_TO_CLASS[kind](fs=fs, **params)


@dataclass(frozen=True)
class ChainBlock:
    """One chain entry: stable id, kind, current params, and validity state.

    `filter` is the constructed `FilterDesign` when `params` are valid for
    the chain's current `fs`; otherwise it is `None` and `error` carries the
    `FilterDesign.validate()` message verbatim (CONTRACTS.md §5: never
    silently clamp -- the invalid params stay exactly as entered, visible on
    the block, until they're fixed).

    `enabled` (default `True`) is a bypass switch, independent of validity: a
    disabled block stays in the chain -- visible, selectable, reorderable,
    still individually inspectable -- but is excluded from every combined/
    exported artifact (`valid_filters`, combined responses, export snapshot,
    C header, PNGs, PDF filter sections) as if it were not there at all. An
    invalid *disabled* block never blocks Export (`has_invalid_blocks` below
    only looks at enabled blocks); only an invalid *enabled* block does.
    """

    id: str
    kind: BlockKind
    params: Mapping[str, float]
    filter: FilterDesign | None
    error: str | None
    enabled: bool = True

    @property
    def is_valid(self) -> bool:
        return self.filter is not None


def _make_block(
    block_id: str, kind: BlockKind, fs: float, params: Mapping[str, float], *, enabled: bool = True
) -> ChainBlock:
    frozen_params = MappingProxyType(dict(params))
    try:
        filt = _build(kind, fs, frozen_params)
        return ChainBlock(id=block_id, kind=kind, params=frozen_params, filter=filt, error=None, enabled=enabled)
    except ValueError as exc:
        return ChainBlock(id=block_id, kind=kind, params=frozen_params, filter=None, error=str(exc), enabled=enabled)


class FilterChain:
    """Qt-independent chain/application model (CONTRACTS.md §12, §13).

    Owns the chain-wide `fs` and the ordered list of filter blocks. No
    plotting, export, persistence, or Qt widget concerns live here -- those
    belong to Phase 5/6 (`ui/*.py`, `export.py`), which are expected to read
    this model and never hold their own copy of filter parameters
    (CONTRACTS.md §12 ownership rule).
    """

    def __init__(self, fs: float) -> None:
        self._validate_fs(fs)
        self._fs = float(fs)
        self._blocks: list[ChainBlock] = []
        self._next_id = itertools.count(1)
        self._selected_id: str | None = None  # None == "Combined" view
        self.dirty: bool = False

    @staticmethod
    def _validate_fs(fs: float) -> None:
        if not (FS_MIN_HZ <= fs <= FS_MAX_HZ):
            raise ValueError(f"fs must be in [{FS_MIN_HZ}, {FS_MAX_HZ}] Hz, got {fs}")

    # -- fs -------------------------------------------------------------

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        self._validate_fs(value)
        self._fs = float(value)
        self._blocks = [_make_block(b.id, b.kind, self._fs, b.params, enabled=b.enabled) for b in self._blocks]
        self.dirty = True

    # -- blocks -----------------------------------------------------------

    @property
    def blocks(self) -> list[ChainBlock]:
        """Ordered snapshot of the chain's blocks. Mutate via chain methods, not this list."""
        return list(self._blocks)

    @property
    def valid_filters(self) -> list[FilterDesign]:
        """Ordered `FilterDesign`s for currently-enabled, currently-valid blocks.

        Feeds `error_analysis.combined_response_error(chain.valid_filters, backend)`
        directly -- error_analysis.py (Phase 3) takes a plain
        `Sequence[FilterDesign]` and knows nothing about `ChainBlock`/ids. A
        disabled block is a bypass and is excluded here exactly like an
        invalid one, so every combined response (ideal, Q14, error metrics)
        automatically treats it as absent from the chain.
        """
        return [b.filter for b in self._blocks if b.enabled and b.filter is not None]

    def get_block(self, block_id: str) -> ChainBlock:
        for b in self._blocks:
            if b.id == block_id:
                return b
        raise KeyError(f"no block with id {block_id!r}")

    def _index_of(self, block_id: str) -> int:
        for i, b in enumerate(self._blocks):
            if b.id == block_id:
                return i
        raise KeyError(f"no block with id {block_id!r}")

    # -- mutation -----------------------------------------------------------

    def add_block(self, kind: BlockKind, **params: float) -> str:
        """Appends a new block, returning its stable id.

        `params` override `DEFAULT_PARAMS[kind]` per-key; omitted params use
        the documented default. Invalid params (e.g. an explicit
        out-of-range override) do not raise -- the block is still added,
        marked invalid, with `.error` set (see module docstring).
        """
        if kind not in _KIND_TO_CLASS:
            raise ValueError(f"unknown block kind {kind!r}, expected one of {sorted(_KIND_TO_CLASS)}")
        resolved = dict(DEFAULT_PARAMS[kind])
        resolved.update(params)
        block_id = f"blk{next(self._next_id)}"
        self._blocks.append(_make_block(block_id, kind, self._fs, resolved))
        self.dirty = True
        return block_id

    def remove_block(self, block_id: str) -> None:
        idx = self._index_of(block_id)
        del self._blocks[idx]
        if self._selected_id == block_id:
            self._selected_id = None  # fall back to Combined, CONTRACTS.md §12
        self.dirty = True

    def move_block(self, block_id: str, to_index: int) -> None:
        """Reorders `block_id` to `to_index` (clamped to the valid range). Id is unchanged."""
        idx = self._index_of(block_id)
        block = self._blocks.pop(idx)
        to_index = max(0, min(to_index, len(self._blocks)))
        self._blocks.insert(to_index, block)
        self.dirty = True

    def update_params(self, block_id: str, **params: float) -> None:
        """Merges `params` into the block's current params and re-validates.

        Never raises for an invalid result (see module docstring) -- check
        `chain.get_block(block_id).is_valid` / `.error` afterward.
        """
        idx = self._index_of(block_id)
        old = self._blocks[idx]
        merged = dict(old.params)
        merged.update(params)
        self._blocks[idx] = _make_block(old.id, old.kind, self._fs, merged, enabled=old.enabled)
        self.dirty = True

    def set_enabled(self, block_id: str, enabled: bool) -> None:
        """Toggles a block's bypass state without touching its params/validity.

        A disabled block stays exactly where it is in the chain, with its
        params and `.error`/`.filter` untouched -- only `.enabled` flips, via
        `dataclasses.replace` rather than re-running `_make_block()` (no
        re-validation needed: enabling/disabling can never change whether a
        block's own params are valid).
        """
        idx = self._index_of(block_id)
        old = self._blocks[idx]
        if old.enabled == enabled:
            return
        self._blocks[idx] = replace(old, enabled=enabled)
        self.dirty = True

    def clear(self) -> None:
        """Empties the block list. `fs` is left unchanged (CONTRACTS.md §13)."""
        self._blocks = []
        self._selected_id = None
        self.dirty = True

    # -- selection ------------------------------------------------------------

    @property
    def selected_id(self) -> str | None:
        """`None` means the "Combined" view is selected."""
        return self._selected_id

    def select(self, block_id: str | None) -> None:
        if block_id is not None:
            self._index_of(block_id)  # raises KeyError if unknown
        self._selected_id = block_id

    @property
    def selected_block(self) -> ChainBlock | None:
        if self._selected_id is None:
            return None
        return self.get_block(self._selected_id)

    # -- dirty state ------------------------------------------------------------

    def mark_clean(self) -> None:
        """Call after an explicit Validate pass (CONTRACTS.md §13); selection is not an edit."""
        self.dirty = False

    # -- validity ------------------------------------------------------------

    @property
    def has_invalid_blocks(self) -> bool:
        """True iff any *enabled* block is invalid.

        A disabled block's invalidity is irrelevant here -- it is bypassed
        out of every combined/exported artifact regardless, so it must never
        gate Export or the "chain has a problem" toolbar state (see
        `ChainBlock.enabled`'s docstring).
        """
        return any(b.enabled and not b.is_valid for b in self._blocks)

    def validation_messages(self) -> dict[str, str]:
        """`{block_id: error_message}` for every currently-invalid block."""
        return {b.id: b.error for b in self._blocks if b.error is not None}

    # -- combined responses ------------------------------------------------------------

    def combined_ideal_response(self, freq_hz: np.ndarray) -> FrequencyResponse:
        """Series-cascade ideal response (CONTRACTS.md §12: dB-additive, phase unwrapped after sum)."""
        return self._combined_response(freq_hz, backend=None, use_q14=False)

    def combined_q14_response(self, freq_hz: np.ndarray, backend: NativeBackend) -> FrequencyResponse:
        """Series-cascade Q14 response. Same combination rule as `combined_ideal_response`."""
        if backend is None:
            raise ValueError("a NativeBackend is required for Q14 comparisons (none was provided)")
        return self._combined_response(freq_hz, backend=backend, use_q14=True)

    def _combined_response(
        self, freq_hz: np.ndarray, backend: NativeBackend | None, use_q14: bool
    ) -> FrequencyResponse:
        if not self._blocks:
            raise ValueError("cannot compute a combined response: filter chain is empty")
        valid = self.valid_filters
        if not valid:
            raise ValueError(
                f"cannot compute a combined response: all {len(self._blocks)} block(s) "
                "in the chain are invalid or disabled"
            )
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        magnitude_db = np.zeros_like(freq_hz)
        phase_deg = np.zeros_like(freq_hz)
        for filt in valid:
            resp = filt.q14_response(freq_hz, backend) if use_q14 else filt.ideal_response(freq_hz)
            magnitude_db += resp.magnitude_db
            phase_deg += resp.phase_deg
        phase_deg = np.degrees(np.unwrap(np.radians(phase_deg)))
        return FrequencyResponse(freq_hz=freq_hz, magnitude_db=magnitude_db, phase_deg=phase_deg)
