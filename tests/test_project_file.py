"""Round-trip and error-handling tests for project_file.py (CONTRACTS.md §15)."""

from __future__ import annotations

import json

import pytest

from filters.chain import FilterChain
from project_file import ProjectFileError, SCHEMA_VERSION, load_project, save_project


def test_round_trip_preserves_fs_and_all_block_kinds(tmp_path):
    chain = FilterChain(fs=22_050.0)
    chain.add_block("LP", fc=3_000.0)
    chain.add_block("HP", fc=500.0)
    bp_id = chain.add_block("BP", f_low=1_000.0, f_high=2_000.0)
    chain.add_block("AP", fc=1_500.0, Q=1.2)
    chain.add_block("PK", fc=2_500.0, Q=1.0, gain_db=6.0)
    chain.set_enabled(bp_id, False)

    path = tmp_path / "chain.iirfilt"
    save_project(chain, path)
    fs, blocks = load_project(path)

    assert fs == 22_050.0
    assert [b["kind"] for b in blocks] == ["LP", "HP", "BP", "AP", "PK"]
    assert blocks[0]["params"] == {"fc": 3_000.0}
    assert blocks[2]["params"] == {"f_low": 1_000.0, "f_high": 2_000.0}
    assert blocks[3]["params"] == {"fc": 1_500.0, "Q": 1.2}
    assert blocks[4]["params"] == {"fc": 2_500.0, "Q": 1.0, "gain_db": 6.0}
    assert [b["enabled"] for b in blocks] == [True, True, False, True, True]


def test_round_trip_preserves_invalid_block(tmp_path):
    """A project file is round-trip state, not an export snapshot -- an
    invalid block must survive save/load exactly, params untouched."""
    chain = FilterChain(fs=13_333.0)
    chain.add_block("LP", fc=999_999.0)  # out of range, stored invalid
    assert not chain.blocks[0].is_valid

    path = tmp_path / "chain.iirfilt"
    save_project(chain, path)
    fs, blocks = load_project(path)

    assert blocks[0]["params"] == {"fc": 999_999.0}


def test_saved_file_is_lf_only_utf8(tmp_path):
    chain = FilterChain(fs=13_333.0)
    chain.add_block("LP")
    path = tmp_path / "chain.iirfilt"
    save_project(chain, path)

    raw = path.read_bytes()
    assert b"\r" not in raw
    raw.decode("utf-8")  # must not raise

    data = json.loads(raw.decode("utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION


def test_missing_file_raises_project_file_error(tmp_path):
    with pytest.raises(ProjectFileError, match="failed to read"):
        load_project(tmp_path / "does_not_exist.iirfilt")


def test_malformed_json_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    path.write_text("{ not valid json ", encoding="utf-8")
    with pytest.raises(ProjectFileError, match="not valid JSON"):
        load_project(path)


def test_non_dict_top_level_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ProjectFileError, match="JSON object"):
        load_project(path)


def test_missing_schema_version_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    path.write_text(json.dumps({"fs": 13333.0, "blocks": []}), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="schema_version"):
        load_project(path)


def test_unrecognized_schema_version_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    path.write_text(json.dumps({"schema_version": 999, "fs": 13333.0, "blocks": []}), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="schema_version"):
        load_project(path)


def test_unknown_block_kind_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    data = {"schema_version": SCHEMA_VERSION, "fs": 13333.0, "blocks": [{"kind": "NOTCH", "params": {}, "enabled": True}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="unknown kind"):
        load_project(path)


def test_missing_fs_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "blocks": []}), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="fs"):
        load_project(path)


def test_blocks_not_a_list_raises_project_file_error(tmp_path):
    path = tmp_path / "bad.iirfilt"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "fs": 13333.0, "blocks": {}}), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="'blocks' must be a list"):
        load_project(path)
