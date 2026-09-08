"""Tests for export.py (Phase 6): CONTRACTS.md §10, §13; CONCEPT.md §7.

Uses the session-scoped `native_backend` fixture from conftest.py (a real
ctypes-backed NativeBackend) since header/PDF content depends on real Q14
quantization, not a stand-in. `gcc` is required on PATH for the header
compilation test, matching the rest of this suite's native-backend tests.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from error_analysis import COEFFICIENT_SWEEP_N
from export import (
    C_SRC_DIR,
    FILTER_KIND_NAMES,
    FIRMWARE_FILTER_SOURCES_DIRNAME,
    MAGNITUDE_YLIM_FLOOR,
    MAGNITUDE_YLIM_STEP,
    ExportError,
    _adaptive_magnitude_ylim,
    _sweep_design_at,
    build_snapshot,
    export_design,
    render_header,
)
from filters import FilterChain
from filters.base import fc_max

FS = 13333.0
FIXED_NOW = datetime(2026, 8, 18, 10, 30, 0, tzinfo=timezone(timedelta(hours=2)))


def _chain_lp_hp_bp_ap(fs: float = FS) -> FilterChain:
    chain = FilterChain(fs=fs)
    chain.add_block("LP", fc=3000.0)
    chain.add_block("HP", fc=1000.0)
    chain.add_block("BP", f_low=2000.0, f_high=4000.0)
    chain.add_block("AP", fc=2000.0, Q=1.0)
    return chain


# --- complete export file set -------------------------------------------------


def test_export_produces_complete_file_set(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    assert result.output_dir.parent == tmp_path
    assert result.output_dir.name == "export_20260818_103000"

    names = {p.name for p in result.output_dir.iterdir()}
    assert names == {
        "report.pdf",
        "filter_design.h",
        "bode_combined.png",
        "bode_lp_1.png",
        "bode_hp_2.png",
        "bode_bp_3.png",
        "bode_ap_4.png",
        "error_sweep_1.png",
        "error_sweep_2.png",
        "error_sweep_3.png",  # BP now has a coefficient sweep too (CONTRACTS.md §6.3)
        "error_sweep_4.png",
        "firmware",  # drop-in C package subfolder -- see test_firmware_package.py
    }


def test_export_creates_timestamped_directory_and_handles_collision(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)

    first = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    second = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    assert first.output_dir != second.output_dir
    assert first.output_dir.is_dir()
    assert second.output_dir.is_dir()
    assert second.output_dir.name == "export_20260818_103000_1"


# --- PDF generation -----------------------------------------------------------


def test_pdf_is_generated_and_nonempty(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    assert result.pdf_path.is_file()
    data = result.pdf_path.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 1000


def test_pdf_write_failure_wrapped_as_export_error(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    from export import render_pdf

    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    # A directory in place of the target PDF path makes the underlying
    # write fail with a real OSError, exercising render_pdf's own wrapping.
    bad_path = tmp_path / "report.pdf"
    bad_path.mkdir()
    with pytest.raises(ExportError, match="report.pdf"):
        render_pdf(snapshot, bad_path, tmp_path / "combined.png", {})


# --- adaptive magnitude y-axis floor ---------------------------------------------


def test_adaptive_magnitude_ylim_keeps_top_unchanged():
    _bottom, top = _adaptive_magnitude_ylim(-37.2, 12.6)
    assert top == 12.6  # top bound formula is untouched by the adaptive-floor change


def test_adaptive_magnitude_ylim_rounds_bottom_down_to_clean_step():
    bottom, _top = _adaptive_magnitude_ylim(-37.2, 12.6)
    assert bottom == -40.0  # floor(-37.2 / 10) * 10
    assert bottom % MAGNITUDE_YLIM_STEP == 0


def test_adaptive_magnitude_ylim_caps_bottom_at_floor_for_deep_notches():
    # A filter with an extreme notch/null must not drag the whole axis down
    # past the -100 dB floor and squash everything else into a sliver.
    bottom, top = _adaptive_magnitude_ylim(-250.0, 3.0)
    assert bottom == MAGNITUDE_YLIM_FLOOR
    assert top == 3.0


def test_adaptive_magnitude_ylim_does_not_squash_small_excursion_filters():
    # A Peak/EQ filter with a small, mostly-flat response shouldn't be
    # forced down to the full -100 dB floor -- the point of the adaptive
    # bottom.
    bottom, top = _adaptive_magnitude_ylim(-2.1, 6.4)
    assert bottom > MAGNITUDE_YLIM_FLOOR
    assert bottom == -10.0
    assert top == 6.4


def test_adaptive_magnitude_ylim_guards_against_degenerate_span():
    bottom, top = _adaptive_magnitude_ylim(-150.0, -120.0)  # both below the floor
    assert top == MAGNITUDE_YLIM_FLOOR
    assert bottom == MAGNITUDE_YLIM_FLOOR - MAGNITUDE_YLIM_STEP
    assert bottom < top


# --- PNG generation -------------------------------------------------------------


def test_bode_pngs_are_valid_png_files(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    for path in [result.combined_bode_png, *result.block_bode_pngs.values()]:
        assert path.is_file()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_error_sweep_pngs_generated_for_all_kinds_including_bp(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    assert set(result.error_sweep_pngs) == {1, 2, 3, 4}  # LP, HP, BP, AP -- all four kinds
    for path in result.error_sweep_pngs.values():
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_write_failure_wrapped_as_export_error(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    # Point the export root at a path that already exists as a *file* --
    # directory creation fails, which is the first filesystem operation
    # export_design performs.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    with pytest.raises(ExportError):
        export_design(chain, native_backend, blocked, now=FIXED_NOW)


# --- header content and coefficient ordering ------------------------------------


def test_header_structure_and_pragma_once(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    text = render_header(snapshot)

    lines = text.splitlines()
    assert lines[0] == "#pragma once"
    assert "fs = 13333 Hz" in text
    assert "#define Q14_SCALE 16384" in text
    assert "#define Q14_TO_FLOAT(x) ((float)(x) / Q14_SCALE)" in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_header_is_ascii_and_lf_terminated(native_backend):
    chain = _chain_lp_hp_bp_ap()
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    text = render_header(snapshot)

    text.encode("ascii")  # raises if any non-ASCII character slipped in
    raw = text.encode("ascii")
    assert b"\r" not in raw


def test_header_coefficient_order_is_b0_b1_b2_a1_a2(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    text = render_header(snapshot)

    filt1_lines = [l for l in text.splitlines() if l.startswith("#define FILT1_")]
    order = [l.split()[1].removeprefix("FILT1_") for l in filt1_lines]
    assert order == ["B0", "B1", "B2", "A1", "A2"]


def test_header_values_match_snapshot_coefficients(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    text = render_header(snapshot)
    block = snapshot.blocks[0]

    for coef in ("b0", "b1", "b2", "a1", "a2"):
        int_val = getattr(block.q14_coefficients, coef)
        float_val = getattr(block.ideal_coefficients, coef)
        define_line = next(l for l in text.splitlines() if l.startswith(f"#define FILT1_{coef.upper()} "))
        assert f" {int_val} " in define_line or define_line.split()[2] == str(int_val)
        assert f"{float_val:.8f}f" in define_line


def test_header_numbering_is_position_based_after_reorder(native_backend):
    chain = FilterChain(fs=FS)
    id_lp = chain.add_block("LP", fc=3000.0)
    id_hp = chain.add_block("HP", fc=1000.0)
    chain.move_block(id_hp, 0)  # HP now first, LP second

    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    text = render_header(snapshot)

    assert "// Filter 1: Butterworth High-Pass" in text
    assert "// Filter 2: Butterworth Low-Pass" in text
    assert "FILT1_B0" in text and "FILT2_B0" in text

    # Stable chain ids must never leak into the exported header text.
    assert id_lp not in text
    assert id_hp not in text


# --- header compilation with GCC ------------------------------------------------


def test_generated_header_compiles_with_gcc(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    stub = tmp_path / "stub.c"
    stub.write_text(
        '#include "filter_design.h"\n'
        '#include "filter_design.h"\n'  # exercise the #pragma once guard
        "int main(void) {\n"
        "    return (int)(FILT1_B0 + FILT2_A1 + FILT3_B2 + FILT4_A2 "
        "+ Q14_TO_FLOAT(FILT1_A1));\n"
        "}\n"
    )
    proc = subprocess.run(
        ["gcc", "-Wall", "-Wextra", "-Werror", "-std=c11", "-I", str(result.output_dir), "-c", str(stub), "-o", str(tmp_path / "stub.o")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


# --- forced fresh validation -----------------------------------------------------


def test_export_reflects_live_chain_state_not_a_stale_cache(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)

    chain.update_params(bid, fc=5000.0)
    # No Validate step is ever run here -- export must compute fresh metrics
    # directly from the current live params, per CONTRACTS.md §13.
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    live_filt = chain.get_block(bid).filter
    assert snapshot.blocks[0].ideal_coefficients == live_filt.ideal_coefficients()
    assert snapshot.blocks[0].params["fc"] == 5000.0


def test_export_recomputes_after_param_change_between_two_exports(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)

    snap1 = build_snapshot(chain, native_backend, now=FIXED_NOW)
    chain.update_params(bid, fc=5000.0)
    snap2 = build_snapshot(chain, native_backend, now=FIXED_NOW)

    assert snap1.blocks[0].ideal_coefficients != snap2.blocks[0].ideal_coefficients
    assert snap2.blocks[0].params["fc"] == 5000.0


# --- BP coefficient sweep (CONTRACTS.md §6.3) --------------------------------------


def test_bp_block_has_coefficient_sweep_defined(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("BP", f_low=2000.0, f_high=4000.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    block = snapshot.blocks[0]
    assert block.coefficient_sweep is not None
    assert block.sweep_unavailable_reason is None
    assert block.coefficient_sweep.max_abs >= 0.0
    assert 100.0 <= block.coefficient_sweep.worst_frequency_hz <= fc_max(FS)
    # Response error is unaffected either way and must still be present.
    assert block.response_error is not None


def test_bp_sweep_design_factory_covers_all_1000_points(native_backend):
    """CONTRACTS.md §6.3: the sweep is linspace(100, fc_max(fs), 1000) for every kind."""
    chain = FilterChain(fs=FS)
    bid = chain.add_block("BP", f_low=2000.0, f_high=4000.0)
    block = chain.get_block(bid)
    design_at = _sweep_design_at(block, chain.fs)
    assert design_at is not None

    calls: list[float] = []

    def counting_design_at(x):
        calls.append(x)
        return design_at(x)

    from error_analysis import coefficient_sweep

    coefficient_sweep(chain.fs, counting_design_at, native_backend)

    assert len(calls) == COEFFICIENT_SWEEP_N == 1000
    assert calls[0] == pytest.approx(100.0)
    assert calls[-1] == pytest.approx(fc_max(chain.fs))


def test_bp_sweep_design_factory_clamps_wide_bandwidth_without_raising(native_backend):
    """A BP block whose bandwidth spans nearly the full domain must still sweep
    cleanly: the clamp keeps f_low < f_high at every point, even at the edges."""
    chain = FilterChain(fs=FS)
    hi = fc_max(FS)
    bid = chain.add_block("BP", f_low=100.0, f_high=hi)  # widest possible bandwidth
    block = chain.get_block(bid)
    design_at = _sweep_design_at(block, chain.fs)

    from error_analysis import coefficient_sweep

    result = coefficient_sweep(chain.fs, design_at, native_backend)  # must not raise
    assert result.max_abs >= 0.0


def test_error_sweep_png_generated_for_bp(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("BP", f_low=2000.0, f_high=4000.0)
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    assert 1 in result.error_sweep_pngs
    assert result.error_sweep_pngs[1].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_bp_sweep_results_appear_in_pdf(tmp_path, native_backend, monkeypatch):
    """Proves BP's sweep numbers reach the PDF story, not just the snapshot --
    parses the Paragraph text fed to reportlab rather than the compressed PDF
    bytes."""
    chain = _chain_lp_hp_bp_ap()
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    bp_block = next(b for b in snapshot.blocks if b.kind == "BP")
    assert bp_block.coefficient_sweep is not None

    from export import Paragraph as _RealParagraph

    captured: list[str] = []

    def fake_paragraph(text, style):
        captured.append(text)
        return _RealParagraph(text, style)

    monkeypatch.setattr("export.Paragraph", fake_paragraph)

    from export import render_pdf

    render_pdf(snapshot, tmp_path / "report.pdf", tmp_path / "combined.png", {})

    expected_fragment = f"Coefficient sweep -- max abs: {bp_block.coefficient_sweep.max_abs:.3e}"
    assert any(expected_fragment in text for text in captured)


def test_all_block_kinds_have_sweep_defined(native_backend):
    """LP/HP/AP (pre-existing), BP (this fix), and PK all get a coefficient sweep."""
    chain = _chain_lp_hp_bp_ap()
    chain.add_block("PK", fc=2500.0, Q=1.0, gain_db=6.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    assert {b.kind for b in snapshot.blocks} == {"LP", "HP", "BP", "AP", "PK"}
    for block in snapshot.blocks:
        assert block.coefficient_sweep is not None
        assert block.sweep_unavailable_reason is None


def test_peak_block_description_and_kind_name(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("PK", fc=2500.0, Q=1.0, gain_db=6.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    assert FILTER_KIND_NAMES["PK"] == "Peaking EQ"
    header_text = render_header(snapshot)
    assert "Peaking EQ" in header_text
    assert "fc = 2500 Hz  Q = 1  Gain = +6 dB" in header_text


# --- invalid-chain rejection -------------------------------------------------------


def test_export_rejects_chain_with_invalid_block(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    chain.add_block("LP", fc=999_999.0)  # invalid: exceeds fc_max

    with pytest.raises(ValueError, match="invalid"):
        export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    with pytest.raises(ValueError, match="invalid"):
        build_snapshot(chain, native_backend, now=FIXED_NOW)

    assert not any(tmp_path.iterdir())  # nothing written -- rejected before filesystem I/O


def test_export_rejects_chain_invalidated_by_fs_change(tmp_path, native_backend):
    chain = FilterChain(fs=40_000.0)
    chain.add_block("LP", fc=15_000.0)  # valid at fs=40_000
    chain.fs = 5_000.0  # now invalid: fc_max(5_000) = 2_250

    with pytest.raises(ValueError, match="invalid"):
        export_design(chain, native_backend, tmp_path, now=FIXED_NOW)


def test_export_still_succeeds_for_a_failing_but_valid_design(tmp_path, native_backend):
    # CONTRACTS.md §13: export is not blocked by a failing 0.1 dB badge --
    # only by invalid *parameters*. A valid design that happens to exceed
    # the threshold must still export, with FAIL recorded, not raise.
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    assert result.snapshot.blocks[0].response_passed in (True, False)  # always computed, never blocks export
    assert result.pdf_path.is_file()


# --- empty-chain behavior -----------------------------------------------------------


def test_export_rejects_empty_chain(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    with pytest.raises(ValueError, match="empty"):
        export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    with pytest.raises(ValueError, match="empty"):
        build_snapshot(chain, native_backend, now=FIXED_NOW)

    assert not any(tmp_path.iterdir())


def test_export_missing_backend_raises(tmp_path):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    with pytest.raises(ValueError, match="NativeBackend"):
        build_snapshot(chain, None, now=FIXED_NOW)


# --- disabled blocks (bypass) are excluded from every export artifact -------------


def test_export_rejects_chain_with_no_enabled_blocks(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=3000.0)
    chain.set_enabled(bid, False)

    with pytest.raises(ValueError, match="no active"):
        build_snapshot(chain, native_backend, now=FIXED_NOW)
    with pytest.raises(ValueError, match="no active"):
        export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    assert not any(tmp_path.iterdir())


def test_export_ignores_invalid_disabled_block(tmp_path, native_backend):
    """An invalid block must not block export once it is disabled (bypass)."""
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    bad_id = chain.add_block("LP", fc=999_999.0)  # invalid
    chain.set_enabled(bad_id, False)

    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    assert result.pdf_path.is_file()
    assert len(result.snapshot.blocks) == 1


def test_disabled_block_excluded_from_snapshot_and_renumbered(native_backend):
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP", fc=3000.0)
    id2 = chain.add_block("HP", fc=1000.0)
    chain.add_block("AP", fc=2000.0, Q=1.0)
    chain.set_enabled(id2, False)  # disable the middle block

    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    assert [b.kind for b in snapshot.blocks] == ["LP", "AP"]
    assert [b.position for b in snapshot.blocks] == [1, 2]  # contiguous over the active list
    assert [b.name for b in snapshot.blocks] == ["FILT1", "FILT2"]
    assert snapshot.blocks[0].block_id == id1


def test_disabled_block_excluded_from_header(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    bid = chain.add_block("HP", fc=1000.0)
    chain.set_enabled(bid, False)

    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    text = render_header(snapshot)

    assert "Butterworth High-Pass" not in text
    assert "FILT2" not in text
    assert "// Filter 1: Butterworth Low-Pass" in text


def test_disabled_block_excluded_from_pngs_and_file_set(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    hp_id = chain.blocks[1].id
    chain.set_enabled(hp_id, False)

    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    names = {p.name for p in result.output_dir.iterdir()}
    assert "bode_hp_2.png" not in names
    # LP/BP/AP renumbered contiguously over the 3 remaining active blocks.
    assert names == {
        "report.pdf",
        "filter_design.h",
        "bode_combined.png",
        "bode_lp_1.png",
        "bode_bp_2.png",
        "bode_ap_3.png",
        "error_sweep_1.png",
        "error_sweep_2.png",
        "error_sweep_3.png",
        "firmware",
    }


def test_disabled_block_excluded_from_combined_response(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    bid = chain.add_block("HP", fc=1000.0)
    chain.set_enabled(bid, False)

    snapshot_with_disabled = build_snapshot(chain, native_backend, now=FIXED_NOW)

    only_lp = FilterChain(fs=FS)
    only_lp.add_block("LP", fc=3000.0)
    snapshot_only_lp = build_snapshot(only_lp, native_backend, now=FIXED_NOW)

    assert snapshot_with_disabled.combined_response_error.max_db == pytest.approx(
        snapshot_only_lp.combined_response_error.max_db
    )


def test_disabled_block_excluded_from_pdf_story(tmp_path, native_backend, monkeypatch):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    bid = chain.add_block("HP", fc=1000.0)
    chain.set_enabled(bid, False)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    from export import Paragraph as _RealParagraph

    captured: list[str] = []

    def fake_paragraph(text, style):
        captured.append(text)
        return _RealParagraph(text, style)

    monkeypatch.setattr("export.Paragraph", fake_paragraph)
    from export import render_pdf

    render_pdf(snapshot, tmp_path / "report.pdf", tmp_path / "combined.png", {})

    assert not any("High-Pass" in text for text in captured)
    assert any("Low-Pass" in text for text in captured)


# --- PDF per-filter page breaks --------------------------------------------------


def _story_for(chain, native_backend) -> list:
    """Captures the `story` list passed to `SimpleDocTemplate.build()` without writing a real PDF."""
    import export

    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)
    captured: dict[str, list] = {}

    class _CapturingDoc:
        def __init__(self, *a, **k):
            pass

        def build(self, story):
            captured["story"] = story

    orig = export.SimpleDocTemplate
    export.SimpleDocTemplate = _CapturingDoc
    try:
        export.render_pdf(snapshot, Path("unused.pdf"), Path("unused_combined.png"), {})
    finally:
        export.SimpleDocTemplate = orig
    return captured["story"]


def test_pdf_inserts_page_break_before_every_filter_section(native_backend):
    from reportlab.platypus import PageBreak, Paragraph

    chain = _chain_lp_hp_bp_ap()
    story = _story_for(chain, native_backend)

    heading_indices = [
        i for i, item in enumerate(story) if isinstance(item, Paragraph) and item.text.startswith("FILT")
    ]
    assert len(heading_indices) == 4  # LP, HP, BP, AP
    for idx in heading_indices:
        assert isinstance(story[idx - 1], PageBreak)


def test_pdf_summary_has_no_leading_page_break(native_backend):
    from reportlab.platypus import PageBreak

    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    story = _story_for(chain, native_backend)

    first_break = next(i for i, item in enumerate(story) if isinstance(item, PageBreak))
    assert first_break > 0  # the title/summary paragraphs precede it
    assert not any(isinstance(item, PageBreak) for item in story[:first_break])


def test_pdf_disabled_block_gets_no_page_break_or_section(native_backend):
    from reportlab.platypus import PageBreak

    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    bid = chain.add_block("HP", fc=1000.0)
    chain.set_enabled(bid, False)
    story = _story_for(chain, native_backend)

    # Only the LP filter section's break; the Combined section still gets
    # its own separate PageBreak on top of that (checked below), so this
    # asserts specifically that the disabled HP block contributes none.
    filter_and_combined_breaks = sum(1 for item in story if isinstance(item, PageBreak))
    assert filter_and_combined_breaks == 2  # LP section + Combined section


def test_pdf_inserts_page_break_before_combined_section(native_backend):
    from reportlab.platypus import PageBreak, Paragraph

    chain = _chain_lp_hp_bp_ap()
    story = _story_for(chain, native_backend)

    combined_idx = next(
        i for i, item in enumerate(story) if isinstance(item, Paragraph) and item.text == "Combined chain"
    )
    assert isinstance(story[combined_idx - 1], PageBreak)


def test_pdf_combined_section_break_survives_with_a_single_block(native_backend):
    """Even a one-block chain (fewest possible per-filter breaks) still gives
    Combined its own leading PageBreak, distinct from the filter section's."""
    from reportlab.platypus import PageBreak, Paragraph

    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    story = _story_for(chain, native_backend)

    assert sum(1 for item in story if isinstance(item, PageBreak)) == 2  # FILT1 section + Combined
    combined_idx = next(
        i for i, item in enumerate(story) if isinstance(item, Paragraph) and item.text == "Combined chain"
    )
    assert isinstance(story[combined_idx - 1], PageBreak)


# --- PDF sub-chapter heading/content keep-together --------------------------------


def test_pdf_headings_are_flagged_keep_with_next(native_backend):
    """Every sub-chapter heading (per-filter, Combined chain, C header
    listing, Test results) must set reportlab's `keepWithNext` so the
    doctemplate's own flowable engine never renders the heading alone at the
    bottom of a page with its content starting on the next (handled via
    keepWithNext rather than an explicit KeepTogether wrapper specifically
    so the flat `story` list -- and the PageBreak-adjacency tests above that
    walk it -- stays unaffected)."""
    from reportlab.platypus import Paragraph

    chain = _chain_lp_hp_bp_ap()
    story = _story_for(chain, native_backend)

    heading_texts = {"Combined chain", "C header listing (filter_design.h)", "Test results"}
    headings = [
        item
        for item in story
        if isinstance(item, Paragraph) and (item.text.startswith("FILT") or item.text in heading_texts)
    ]
    assert len(headings) == 4 + 3  # 4 filter sections + Combined/header-listing/results
    for heading in headings:
        assert heading.getKeepWithNext() == 1

    # The title/summary paragraphs on page 1 are not sub-chapter headings.
    title = next(item for item in story if isinstance(item, Paragraph) and "Export Report" in item.text)
    assert title.getKeepWithNext() == 0


def test_pdf_sections_without_leading_page_break_still_keep_heading_with_content(native_backend):
    """Sections 4 (C header listing) and 5 (Test results) get no forced
    PageBreak of their own (unlike every per-filter section and Combined
    chain) -- keepWithNext is what protects *these* headings specifically
    from landing alone at the bottom of whatever page precedes them."""
    from reportlab.platypus import PageBreak, Paragraph

    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    story = _story_for(chain, native_backend)

    for text in ("C header listing (filter_design.h)", "Test results"):
        idx = next(i for i, item in enumerate(story) if isinstance(item, Paragraph) and item.text == text)
        assert not isinstance(story[idx - 1], PageBreak)
        assert story[idx].getKeepWithNext() == 1


# --- filesystem failure handling -----------------------------------------------------


def test_export_root_colliding_with_existing_file_raises_export_error(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)

    blocked_root = tmp_path / "output"
    blocked_root.write_text("i am a file, not a directory")

    with pytest.raises(ExportError, match="output"):
        export_design(chain, native_backend, blocked_root, now=FIXED_NOW)


def test_export_root_collision_error_has_no_output_dir(tmp_path, native_backend):
    """No export directory was ever created for this failure (it fails
    before `_make_export_dir` can succeed), so `output_dir` stays None --
    the ui/app.py error dialog falls back to the destination the user
    picked in that case."""
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)

    blocked_root = tmp_path / "output"
    blocked_root.write_text("i am a file, not a directory")

    with pytest.raises(ExportError) as excinfo:
        export_design(chain, native_backend, blocked_root, now=FIXED_NOW)
    assert excinfo.value.output_dir is None


def test_export_error_after_dir_creation_carries_the_export_dir(tmp_path, native_backend, monkeypatch):
    """A failure raised *after* the export directory was created (here,
    forced during PDF generation) must have that directory attached to the
    raised ExportError, even though the failure site itself (render_pdf)
    never learns the directory -- export_design() back-fills it (CONTRACTS.md
    §10/§13-adjacent UX: the ui/app.py error dialog can then point at the
    partial export directory instead of only the destination root)."""
    import export as export_module

    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)

    def _boom(*args, **kwargs):
        raise ExportError("boom")

    monkeypatch.setattr(export_module, "render_pdf", _boom)

    with pytest.raises(ExportError) as excinfo:
        export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    assert excinfo.value.output_dir is not None
    assert excinfo.value.output_dir.parent == tmp_path
    assert excinfo.value.output_dir.is_dir()  # the partial export dir really was created


def test_header_write_failure_raises_export_error(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    snapshot = build_snapshot(chain, native_backend, now=FIXED_NOW)

    from export import _write_header

    # A directory in place of the header path forces `open(path, "wb")` to
    # fail with a real OSError (IsADirectoryError on Linux).
    bad_path = tmp_path / "filter_design.h"
    bad_path.mkdir()
    with pytest.raises(ExportError, match="filter_design.h"):
        _write_header(bad_path, snapshot)


def test_png_write_failure_raises_export_error_directly(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=3000.0)
    from error_analysis import bode_grid
    from export import _save_bode_png

    filt = chain.get_block(bid).filter
    freq = bode_grid(chain.fs)
    ideal = filt.ideal_response(freq)
    q14 = filt.q14_response(freq, native_backend)

    bad_path = tmp_path / "somewhere" / "plot.png"  # parent dir doesn't exist
    with pytest.raises(ExportError, match="plot.png"):
        _save_bode_png(bad_path, ideal, q14, "title")


# --- firmware/ drop-in package (CONTRACTS.md §10) -------------------------------------


def test_firmware_subfolder_contains_expected_files(tmp_path, native_backend):
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    assert result.firmware_dir == result.output_dir / "firmware"
    # Top level: only the two files an integrator actually names/reads, plus
    # the README -- every filter *source* file is nested under biquad_q14/.
    names = {p.name for p in result.firmware_dir.iterdir()}
    assert names == {
        "filter_design.h",
        "example.c",
        "README.md",
        FIRMWARE_FILTER_SOURCES_DIRNAME,
    }

    sources_dir = result.firmware_dir / FIRMWARE_FILTER_SOURCES_DIRNAME
    assert sources_dir.is_dir()
    source_names = {p.name for p in sources_dir.iterdir()}
    assert source_names == {
        "filter_design_calc.h",
        "filter_design_calc.c",
        "biquad_q14.h",
        "biquad_q14.c",
    }


def test_firmware_biquad_c_is_byte_identical_to_src_c(tmp_path, native_backend):
    """The DSP logic is copied verbatim, never hand-duplicated -- a Feature-A-style
    change to src/c/biquad_q14.c must propagate to the next export automatically."""
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    bundled = result.firmware_dir / FIRMWARE_FILTER_SOURCES_DIRNAME / "biquad_q14.c"
    assert bundled.read_bytes() == (C_SRC_DIR / "biquad_q14.c").read_bytes()


def test_firmware_design_calc_header_is_byte_identical_to_src_c(tmp_path, native_backend):
    """filter_design_calc.h has no #include of its own, so unlike the paired
    .c file it needs no rewrite -- a pure verbatim copy of src/c/filter_design.h."""
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    bundled = result.firmware_dir / FIRMWARE_FILTER_SOURCES_DIRNAME / "filter_design_calc.h"
    assert bundled.read_bytes() == (C_SRC_DIR / "filter_design.h").read_bytes()


def test_firmware_design_calc_source_matches_src_c_except_include(tmp_path, native_backend):
    """filter_design_calc.c is a verbatim copy of src/c/filter_design.c except
    its own #include line is repointed at the renamed header -- never
    hand-duplicated design math."""
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    original = (C_SRC_DIR / "filter_design.c").read_text(encoding="utf-8")
    bundled = (result.firmware_dir / FIRMWARE_FILTER_SOURCES_DIRNAME / "filter_design_calc.c").read_text(
        encoding="utf-8"
    )
    assert bundled == original.replace('#include "filter_design.h"', '#include "filter_design_calc.h"', 1)


def test_firmware_header_has_no_filter_design_include_and_defines_q14_coeffs(tmp_path, native_backend):
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    text = (result.firmware_dir / FIRMWARE_FILTER_SOURCES_DIRNAME / "biquad_q14.h").read_text(encoding="utf-8")
    assert '#include "filter_design.h"' not in text
    assert '#include "filter_design_calc.h"' in text


def test_firmware_example_includes_point_into_sources_subfolder(tmp_path, native_backend):
    """example.c sits at firmware/'s top level while the headers it needs
    live under biquad_q14/, so its own #includes must be subfolder-qualified
    (CONTRACTS.md §10's firmware/ layout)."""
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    text = (result.firmware_dir / "example.c").read_text(encoding="utf-8")
    assert f'#include "{FIRMWARE_FILTER_SOURCES_DIRNAME}/filter_design_calc.h"' in text
    assert f'#include "{FIRMWARE_FILTER_SOURCES_DIRNAME}/biquad_q14.h"' in text


def test_firmware_filter_design_h_matches_top_level_content(tmp_path, native_backend):
    result = export_design(_chain_lp_hp_bp_ap(), native_backend, tmp_path, now=FIXED_NOW)

    top_level = (result.output_dir / "filter_design.h").read_bytes()
    firmware = (result.firmware_dir / "filter_design.h").read_bytes()
    assert top_level == firmware


def test_firmware_example_declares_one_state_per_active_block(tmp_path, native_backend):
    chain = _chain_lp_hp_bp_ap()
    hp_id = chain.blocks[1].id
    chain.set_enabled(hp_id, False)  # LP, BP, AP remain active -> renumbered FILT1/2/3

    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    text = (result.firmware_dir / "example.c").read_text(encoding="utf-8")

    assert text.count("static biquad_q14_state_t state") == 3
    assert "state1" in text and "state2" in text and "state3" in text
    assert "state4" not in text and "coeffs4" not in text
    assert text.count("biquad_q14_process(&state") == 3


def test_firmware_example_computes_coefficients_via_design_functions(tmp_path, native_backend):
    """example.c must call this design's own filter_design_lp/hp/bp/ap/pk()
    at runtime to obtain each stage's coefficients (CONTRACTS.md §10) --
    never read the frozen FILT<n>_* defines directly, and never hand-compute
    or hardcode a coefficient literal."""
    chain = _chain_lp_hp_bp_ap()  # LP fc=3000, HP fc=1000, BP f_low=2000/f_high=4000, AP fc=2000 Q=1.0
    chain.add_block("PK", fc=2500.0, Q=1.0, gain_db=6.0)
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    text = (result.firmware_dir / "example.c").read_text(encoding="utf-8")

    assert not re.search(r"FILT\d+_", text)  # no frozen-define usage (banner comment may still mention them)
    assert f'#include "{FIRMWARE_FILTER_SOURCES_DIRNAME}/filter_design_calc.h"' in text
    assert re.search(r"filter_design_lp\(\s*3000\.0\s*,\s*13333\.0\s*,\s*&coeffs1\s*\);", text)
    assert re.search(r"filter_design_hp\(\s*1000\.0\s*,\s*13333\.0\s*,\s*&coeffs2\s*\);", text)
    assert re.search(r"filter_design_bp\(\s*2000\.0\s*,\s*4000\.0\s*,\s*13333\.0\s*,\s*&coeffs3\s*\);", text)
    assert re.search(r"filter_design_ap\(\s*2000\.0\s*,\s*13333\.0\s*,\s*1\.0\s*,\s*&coeffs4\s*\);", text)
    assert re.search(r"filter_design_pk\(\s*2500\.0\s*,\s*13333\.0\s*,\s*1\.0\s*,\s*6\.0\s*,\s*&coeffs5\s*\);", text)
