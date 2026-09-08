"""Project file save/open (docs/CONTRACTS.md §15).

A project file captures the full round-trip state of a `FilterChain` (`fs`
plus every block's kind/params/enabled, including invalid or disabled
blocks) as versioned JSON -- distinct from `export.py`'s `ExportSnapshot`,
which is an active-only, freshly-validated deliverable, not a round-trip
format. No UI chrome (splitter sizes, selected tab, window geometry) is
persisted; this is model-only state.

Schema version 2 additionally captures the `SignalChain` (docs/CONCEPT.md
§11): every signal block's kind/params/factor, including invalid blocks
(mirroring the filter-block round-trip policy above) -- there is no
"enabled" flag for signal blocks (§11.2 has none). A `NOISE` block's `seed`
lives inside its own `params` dict (it's just another constructor argument
of `NoiseSignal`, see signals/chain.py's `add_block()`), so it round-trips
for free without a dedicated field. A version-1 file (no `signals` key
at all) is still readable -- it just loads with an empty signal chain --
so opening a project saved before this feature existed never fails.
`file_path` for a `CSV` block is stored verbatim (absolute, as delivered by
the file picker) -- not rewritten to be relative to the project file. If
that path no longer resolves at load time, the block round-trips anyway and
comes back marked invalid with `.error` set (`CsvSignal.validate()`'s
own "CSV file not found" message) -- the existing "never silently clamp,
show the error on the block" policy, not a special case here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filters.chain import FilterChain
from signals import SignalChain

SCHEMA_VERSION = 2
# Every schema_version this reader still accepts -- a file written before
# this feature existed (version 1, no `signals` key) stays loadable.
_SUPPORTED_SCHEMA_VERSIONS = {1, 2}
PROJECT_FILE_EXTENSION = ".iirfilt"

_VALID_KINDS = {"LP", "HP", "BP", "AP", "PK"}
_VALID_SIGNAL_KINDS = {"SIN", "DC", "NOISE", "CSV"}


class ProjectFileError(RuntimeError):
    """A project file could not be saved or is malformed/unreadable."""


def save_project(chain: FilterChain, path: Path | str, signal_chain: SignalChain | None = None) -> None:
    """Writes `chain`'s full state (fs + every block, valid or not, enabled
    or not), plus `signal_chain`'s blocks (kind/params/factor, valid or
    not) if one is given, as versioned JSON. `signal_chain=None` (the
    default -- e.g. `export.py`'s own call site when `export_design()` was
    invoked without one) writes an empty `signals` list, not a missing key,
    so every file this writes is a well-formed current-schema file. LF-only,
    UTF-8 -- line endings never depend on platform (same technique as
    export.py's `_write_header`)."""
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fs": chain.fs,
        "blocks": [
            {"kind": block.kind, "params": dict(block.params), "enabled": block.enabled} for block in chain.blocks
        ],
        "signals": [
            {"kind": block.kind, "params": dict(block.params), "factor": block.factor}
            for block in (signal_chain.blocks if signal_chain is not None else [])
        ],
    }
    text = json.dumps(data, indent=2) + "\n"
    try:
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))
    except OSError as exc:
        raise ProjectFileError(f"failed to write project file {path}: {exc}") from exc


def load_project(path: Path | str) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    """Reads and validates a project file, returning `(fs, blocks, signal_blocks)`
    as plain data -- deliberately does NOT construct a `FilterChain`/`SignalChain`
    itself; the caller applies this to existing chain objects (see `src/ui/app.py`'s
    `_on_open`, which mutates the current chains in place rather than swapping in
    new instances).

    `blocks` is a list of `{"kind": str, "params": dict, "enabled": bool}`.
    `signal_blocks` is a list of `{"kind": str, "params": dict, "factor": float}` --
    always `[]` for a version-1 file (no `signals` section, see module docstring).
    Raises `ProjectFileError` for any missing/unreadable/malformed file.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ProjectFileError(f"failed to read project file {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectFileError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ProjectFileError(f"{path}: expected a JSON object at the top level, got {type(data).__name__}")

    version = data.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ProjectFileError(
            f"{path}: unsupported schema_version {version!r} (expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)})"
        )

    if "fs" not in data:
        raise ProjectFileError(f"{path}: missing required field 'fs'")
    try:
        fs = float(data["fs"])
    except (TypeError, ValueError) as exc:
        raise ProjectFileError(f"{path}: 'fs' must be a number, got {data['fs']!r}") from exc

    raw_blocks = data.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ProjectFileError(f"{path}: 'blocks' must be a list, got {type(raw_blocks).__name__}")

    blocks: list[dict[str, Any]] = []
    for i, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, dict):
            raise ProjectFileError(f"{path}: blocks[{i}] must be an object, got {type(raw_block).__name__}")
        kind = raw_block.get("kind")
        if kind not in _VALID_KINDS:
            raise ProjectFileError(f"{path}: blocks[{i}] has unknown kind {kind!r} (expected one of {sorted(_VALID_KINDS)})")
        params = raw_block.get("params")
        if not isinstance(params, dict):
            raise ProjectFileError(f"{path}: blocks[{i}] 'params' must be an object, got {type(params).__name__}")
        enabled = raw_block.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ProjectFileError(f"{path}: blocks[{i}] 'enabled' must be a boolean, got {type(enabled).__name__}")
        blocks.append({"kind": kind, "params": dict(params), "enabled": enabled})

    signal_blocks: list[dict[str, Any]] = []
    if version >= 2:
        raw_signals = data.get("signals")
        if not isinstance(raw_signals, list):
            raise ProjectFileError(f"{path}: 'signals' must be a list, got {type(raw_signals).__name__}")
        for i, raw_signal in enumerate(raw_signals):
            if not isinstance(raw_signal, dict):
                raise ProjectFileError(f"{path}: signals[{i}] must be an object, got {type(raw_signal).__name__}")
            kind = raw_signal.get("kind")
            if kind not in _VALID_SIGNAL_KINDS:
                raise ProjectFileError(
                    f"{path}: signals[{i}] has unknown kind {kind!r} (expected one of {sorted(_VALID_SIGNAL_KINDS)})"
                )
            params = raw_signal.get("params")
            if not isinstance(params, dict):
                raise ProjectFileError(f"{path}: signals[{i}] 'params' must be an object, got {type(params).__name__}")
            factor = raw_signal.get("factor", 1.0)
            try:
                factor = float(factor)
            except (TypeError, ValueError) as exc:
                raise ProjectFileError(f"{path}: signals[{i}] 'factor' must be a number, got {factor!r}") from exc
            signal_blocks.append({"kind": kind, "params": dict(params), "factor": factor})

    return fs, blocks, signal_blocks
