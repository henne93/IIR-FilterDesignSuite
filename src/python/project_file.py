"""Project file save/open (docs/CONTRACTS.md §15).

A project file captures the full round-trip state of a `FilterChain` (`fs`
plus every block's kind/params/enabled, including invalid or disabled
blocks) as versioned JSON -- distinct from `export.py`'s `ExportSnapshot`,
which is an active-only, freshly-validated deliverable, not a round-trip
format. No UI chrome (splitter sizes, selected tab, window geometry) is
persisted; this is model-only state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filters.chain import FilterChain

SCHEMA_VERSION = 1
PROJECT_FILE_EXTENSION = ".iirfilt"

_VALID_KINDS = {"LP", "HP", "BP", "AP", "PK"}


class ProjectFileError(RuntimeError):
    """A project file could not be saved or is malformed/unreadable."""


def save_project(chain: FilterChain, path: Path | str) -> None:
    """Writes `chain`'s full state (fs + every block, valid or not, enabled
    or not) as versioned JSON. LF-only, UTF-8 -- line endings never depend
    on platform (same technique as export.py's `_write_header`)."""
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fs": chain.fs,
        "blocks": [
            {"kind": block.kind, "params": dict(block.params), "enabled": block.enabled} for block in chain.blocks
        ],
    }
    text = json.dumps(data, indent=2) + "\n"
    try:
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))
    except OSError as exc:
        raise ProjectFileError(f"failed to write project file {path}: {exc}") from exc


def load_project(path: Path | str) -> tuple[float, list[dict[str, Any]]]:
    """Reads and validates a project file, returning `(fs, blocks)` as plain
    data -- deliberately does NOT construct a `FilterChain` itself; the
    caller applies this to an existing chain object (see `src/ui/app.py`'s
    `_on_open`, which mutates the current chain in place rather than
    swapping in a new instance).

    `blocks` is a list of `{"kind": str, "params": dict, "enabled": bool}`.
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
    if version != SCHEMA_VERSION:
        raise ProjectFileError(
            f"{path}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION})"
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

    return fs, blocks
