"""Export layer: PDF report + C header + PNG plots (Phase 6, CONTRACTS.md §10, §13).

Produces a timestamped `export_YYYYMMDD_HHMMSS/` directory containing:

- `report.pdf`            -- design summary, per-filter sections, combined
                             chain, inline header listing, test results
                             (CONCEPT.md §7).
- `filter_design.h`       -- CONTRACTS.md §10 format: ASCII, LF-terminated,
                             `#pragma once` guarded, plain integer `#define`
                             literals (no undefined `Q14(...)` macro).
- `bode_combined.png`     -- series-cascade Bode plot.
- `bode_<kind>_<n>.png`   -- one per chain block, `<n>` = 1-indexed chain
                             position (CONCEPT.md §7 naming).
- `error_sweep_<n>.png`   -- coefficient-accuracy sweep plot per block, for
                             all four filter kinds. LP/HP/AP sweep `fc`
                             directly; BP sweeps its center frequency while
                             holding its configured bandwidth fixed, clamped
                             at the `[100, fc_max(fs)]` domain edges
                             (CONTRACTS.md §6.3) -- see `_sweep_design_at()`.
- `firmware/`             -- a complete, self-contained C package (generated
                             coefficients, `biquad_q14.{h,c}`, a generated
                             cascade-wiring `example.c`, `README.md`) that a
                             firmware integrator can copy into another
                             project as-is (CONTRACTS.md §10). See
                             `render_firmware_package()` below.

The top-level `filter_design.h` above is deliberately coefficient-only, unchanged
from the original design (CONCEPT.md §7's export file listing enumerates exactly
the files above it; `biquad_q14.{h,c}` stays firmware reference source in this
repository's own `src/c/` tree, not duplicated at the top level). The `firmware/`
subfolder is a separate, additive output that *does* bundle a generated,
standalone-package variant of `biquad_q14.{h,c}` -- see `render_firmware_package()`,
`README.md` §6, and `tests/test_firmware_package.py`, which proves that package
compiles, links, and runs correctly, fully standalone, with GCC.

Two-stage design:

1. `build_snapshot()` forces a fresh `error_analysis` pass (never reads any
   UI-layer cache, e.g. `Inspector`'s) and freezes the result into immutable
   `ExportSnapshot`/`BlockSnapshot` dataclasses -- CONTRACTS.md §13: "Export
   always forces a fresh Validate pass first, regardless of cached/stale
   state." Blocks are renumbered `FILT1..FILTn` by chain position; each
   block's stable `FilterChain` id is carried on the snapshot for internal
   traceability only and never appears in a filename or generated file
   (position-based names are the export-facing identity; stable ids stay an
   application-state concern, CONTRACTS.md §12).
2. `export_design()` takes that frozen snapshot and writes the file set,
   wrapping filesystem/plotting/PDF failures in `ExportError` so callers get
   an actionable message instead of a bare `OSError` traceback.

Export is blocked only by invalid parameters on an *enabled* block, an empty
chain, or a chain with no enabled block (§13's "only by *invalid*
parameters" carve-out) -- a design that fails the 0.1 dB response check
still exports; the report documents the failure instead of suppressing it.
A disabled block (`ChainBlock.enabled = False`) is a bypass: it is dropped
from the snapshot, header, PNGs, and PDF filter sections entirely, and an
invalid disabled block never blocks export.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from error_analysis import (
    COEFFICIENT_NAMES,
    COEFFICIENT_SWEEP_FLOOR_HZ,
    COEFFICIENT_SWEEP_N,
    CoefficientSweepError,
    ResponseError,
    bode_grid,
    coefficient_sweep,
    combined_response_error,
    response_error,
)
from filters.base import Coefficients, FilterDesign, NativeBackend, Q14Coefficients, fc_max
from filters.chain import BlockKind, ChainBlock, FilterChain

# Reference firmware source (CONTRACTS.md §7, §10) -- render_firmware_package()
# below reads src/c/biquad_q14.{h,c} and src/c/filter_design.h from here.
C_SRC_DIR = Path(__file__).resolve().parent.parent / "c"

# Sample count for the illustrative demo main() in the generated firmware/example.c.
FIRMWARE_EXAMPLE_N_SAMPLES = 32

# CONTRACTS.md §11: response pass/fail badge threshold (max amplitude error).
RESPONSE_PASS_THRESHOLD_DB = 0.1

# Reduced point count for the *visual* error-sweep curve (report.pdf /
# error_sweep_<n>.png) -- distinct from COEFFICIENT_SWEEP_N, which is the
# locked, regression-significant point count used for the summary stats
# (CONTRACTS.md §14/§11) and must not be altered here.
ERROR_SWEEP_PLOT_N = 200

IDEAL_COLOR = "#2980b9"
Q14_COLOR = "#c0392b"

PHASE_YLIM = (-180.0, 180.0)
MAGNITUDE_YLIM_FLOOR = -100.0

FILTER_KIND_NAMES: Mapping[BlockKind, str] = MappingProxyType(
    {
        "LP": "Butterworth Low-Pass",
        "HP": "Butterworth High-Pass",
        "BP": "Butterworth Band-Pass",
        "AP": "Butterworth All-Pass",
        "PK": "Peaking EQ",
    }
)

class ExportError(RuntimeError):
    """Export failed for a reason outside invalid chain state -- filesystem, plotting, or PDF generation."""


@dataclass(frozen=True)
class BlockSnapshot:
    """Immutable, position-numbered snapshot of one chain block at export time."""

    position: int  # 1-indexed, FILT<n>
    block_id: str  # stable FilterChain id -- traceability only, never exported into a file
    kind: BlockKind
    params: Mapping[str, float]
    derived_params: Mapping[str, float]
    ideal_coefficients: Coefficients
    q14_coefficients: Q14Coefficients
    response_error: ResponseError
    response_passed: bool
    coefficient_sweep: CoefficientSweepError | None
    sweep_unavailable_reason: str | None

    @property
    def name(self) -> str:
        return f"FILT{self.position}"


@dataclass(frozen=True)
class ExportSnapshot:
    """Immutable snapshot of the full chain + freshly-computed analysis results."""

    generated_at: datetime
    fs: float
    blocks: tuple[BlockSnapshot, ...]
    combined_response_error: ResponseError
    combined_passed: bool


def _active_blocks(chain: FilterChain) -> list[ChainBlock]:
    """Enabled blocks, in chain order -- everything export-facing is numbered/built from this list.

    A disabled block is a bypass (CONTRACTS.md-adjacent `ChainBlock.enabled`
    docstring): it never gets a `BlockSnapshot`, header entry, or PNG, and
    `FILT<n>` numbering is contiguous over *this* list, not the full chain.
    """
    return [b for b in chain.blocks if b.enabled]


def _sweep_design_at(block: ChainBlock, fs: float) -> Callable[[float], FilterDesign] | None:
    """Factory for `coefficient_sweep()`'s `design_at` (CONTRACTS.md §6.3).

    LP/HP/AP sweep `fc` directly, holding every other param fixed. BP sweeps
    its center frequency while holding its configured bandwidth
    (`f_high - f_low`) fixed, clamping `f_low`/`f_high` at the
    `[COEFFICIENT_SWEEP_FLOOR_HZ, fc_max(fs)]` domain edges whenever a
    symmetric window around the center would otherwise leave that domain --
    the same clamp strategy validated in
    tests/test_error_analysis.py::test_coefficient_sweep_bp_holds_bandwidth_fixed.
    Since a BP block's own bandwidth never exceeds the domain width (its
    `f_low`/`f_high` are already validated into range), at least half the
    original half-bandwidth always survives the clamp, so `f_low < f_high`
    holds at every swept point -- `design_at` never raises.
    Returns `None` only for a block kind with no sweep definition (none
    currently -- `BlockKind` is exhaustive above); kept for forward
    compatibility with a future filter kind that might not have one.
    """
    filt = block.filter
    if block.kind in ("LP", "HP"):
        return lambda x, f=filt: type(f)(fs=fs, fc=x)
    if block.kind == "AP":
        q = block.params["Q"]
        return lambda x, f=filt, q=q: type(f)(fs=fs, fc=x, Q=q)
    if block.kind == "PK":
        q = block.params["Q"]
        gain_db = block.params["gain_db"]
        return lambda x, f=filt, q=q, gain_db=gain_db: type(f)(fs=fs, fc=x, Q=q, gain_db=gain_db)
    if block.kind == "BP":
        hi = fc_max(fs)
        half_bw = (block.params["f_high"] - block.params["f_low"]) / 2.0

        def design_at(center: float, f=filt, half_bw=half_bw, hi=hi) -> FilterDesign:
            f_low = max(COEFFICIENT_SWEEP_FLOOR_HZ, center - half_bw)
            f_high = min(hi, center + half_bw)
            return type(f)(fs=fs, f_low=f_low, f_high=f_high)

        return design_at
    return None  # pragma: no cover -- BlockKind is exhaustive above


def build_snapshot(chain: FilterChain, backend: NativeBackend, *, now: datetime | None = None) -> ExportSnapshot:
    """Forces a fresh validation pass and freezes chain + analysis state for export.

    Raises `ValueError` for the conditions export refuses to proceed past
    (CONTRACTS.md §13): an empty chain, a chain with no *enabled* block, or
    any *enabled* block with invalid parameters. A design that is merely
    *stale* or has a failing response badge is exported anyway -- this
    function always recomputes every metric from the live chain, so there is
    no stale state to inherit. A disabled block is a bypass: it never blocks
    export (even invalid) and is excluded from the snapshot entirely, as if
    it were not in the chain (see `filters.chain.ChainBlock.enabled`).
    """
    if backend is None:
        raise ValueError("a NativeBackend is required to export (Q14 coefficients/metrics are export-mandatory)")
    if not chain.blocks:
        raise ValueError("cannot export: filter chain is empty")
    active = _active_blocks(chain)
    if not active:
        raise ValueError("cannot export: no active (enabled) filter blocks in the chain")
    invalid_active = [b for b in active if not b.is_valid]
    if invalid_active:
        detail = "; ".join(f"{b.id}: {b.error}" for b in invalid_active)
        raise ValueError(f"cannot export: {len(invalid_active)} block(s) have invalid parameters ({detail})")

    generated_at = now if now is not None else datetime.now().astimezone()

    blocks: list[BlockSnapshot] = []
    for position, block in enumerate(_active_blocks(chain), start=1):
        filt = block.filter  # not None: the invalid_active check above guarantees this
        ideal = filt.ideal_coefficients()
        q14 = filt.q14_coefficients(backend)
        resp_err = response_error(filt, backend)

        sweep: CoefficientSweepError | None = None
        sweep_reason: str | None = None
        design_at = _sweep_design_at(block, chain.fs)
        if design_at is None:
            sweep_reason = f"Coefficient sweep unavailable for block kind {block.kind!r}."
        else:
            sweep = coefficient_sweep(chain.fs, design_at, backend)

        blocks.append(
            BlockSnapshot(
                position=position,
                block_id=block.id,
                kind=block.kind,
                params=MappingProxyType(dict(block.params)),
                derived_params=MappingProxyType(filt.derived_params()),
                ideal_coefficients=ideal,
                q14_coefficients=q14,
                response_error=resp_err,
                response_passed=resp_err.max_db <= RESPONSE_PASS_THRESHOLD_DB,
                coefficient_sweep=sweep,
                sweep_unavailable_reason=sweep_reason,
            )
        )

    combined = combined_response_error(chain.valid_filters, backend)
    combined_passed = combined.max_db <= RESPONSE_PASS_THRESHOLD_DB

    return ExportSnapshot(
        generated_at=generated_at,
        fs=chain.fs,
        blocks=tuple(blocks),
        combined_response_error=combined,
        combined_passed=combined_passed,
    )


# -- C header generation (CONTRACTS.md §10) ----------------------------------


def _filter_description(block: BlockSnapshot) -> str:
    if block.kind in ("LP", "HP"):
        return f"fc = {block.params['fc']:g} Hz"
    if block.kind == "BP":
        return f"f_low = {block.params['f_low']:g} Hz  f_high = {block.params['f_high']:g} Hz"
    if block.kind == "AP":
        return f"fc = {block.params['fc']:g} Hz  Q = {block.params['Q']:g}"
    if block.kind == "PK":
        return f"fc = {block.params['fc']:g} Hz  Q = {block.params['Q']:g}  Gain = {block.params['gain_db']:+g} dB"
    raise ValueError(f"unknown filter kind {block.kind!r}")  # pragma: no cover -- BlockKind is exhaustive above


def render_header(snapshot: ExportSnapshot) -> str:
    """Renders `filter_design.h` text per CONTRACTS.md §10.

    ASCII-only: the doc's illustrative example uses a Unicode em dash in the
    banner comment, which would violate the explicit ASCII requirement here
    -- a plain hyphen is used instead. Coefficient order within each filter
    is always b0, b1, b2, a1, a2 (CONTRACTS.md §2).
    """
    lines = [
        "#pragma once",
        f"// Generated by IIR Filter Design Suite - {snapshot.generated_at.isoformat()}",
        f"// fs = {snapshot.fs:g} Hz",
        "",
        "#define Q14_SCALE 16384",
        "#define Q14_TO_FLOAT(x) ((float)(x) / Q14_SCALE)",
        "",
    ]
    for block in snapshot.blocks:
        lines.append(f"// Filter {block.position}: {FILTER_KIND_NAMES[block.kind]}  {_filter_description(block)}")
        for i, coef in enumerate(COEFFICIENT_NAMES):
            int_val = getattr(block.q14_coefficients, coef)
            float_val = getattr(block.ideal_coefficients, coef)
            define_name = f"FILT{block.position}_{coef.upper()}"
            comment = f"{float_val:.8f}f"
            if i == 0:
                comment += " (Q14, scale=16384)"
            lines.append(f"#define {define_name:<12} {str(int_val):<8} // {comment}")
        lines.append("")

    text = "\n".join(lines).rstrip("\n") + "\n"
    text.encode("ascii")  # raises UnicodeEncodeError if this invariant is ever broken
    return text


def _write_text_lf(path: Path, text: str, *, encoding: str = "ascii") -> None:
    """Writes `text` as raw bytes (not text mode) so LF line endings survive
    unconditionally, even when this runs on Windows (CONTRACTS.md §10: "even
    when generated on Windows"). `encoding="ascii"` (the default) matches the
    generated coefficient header's own ASCII requirement; the firmware
    package's other files (copied/derived from src/c/*, which contain
    non-ASCII punctuation in comments) pass encoding="utf-8" instead."""
    try:
        text.encode(encoding)
    except UnicodeEncodeError as exc:
        raise ExportError(f"content for {path} is not valid {encoding}: {exc}") from exc
    try:
        with open(path, "wb") as f:
            f.write(text.encode(encoding))
    except OSError as exc:
        raise ExportError(f"failed to write {path}: {exc}") from exc


def _write_header(path: Path, snapshot: ExportSnapshot) -> None:
    _write_text_lf(path, render_header(snapshot))


# -- Drop-in firmware package (firmware/ subfolder) ----------------------------
#
# Additive to the top-level export files above, which stay byte-for-byte
# unchanged (CONCEPT.md §7's original file listing, CONTRACTS.md §10). This
# subfolder bundles a complete, self-contained C package -- generated
# coefficients, the biquad implementation, and a cascade-wiring example --
# that a firmware integrator can copy into an external project as-is,
# reversing the original "does not bundle biquad_q14.*" decision for this
# new, separate output only. Never hand-duplicates biquad_q14.c's DSP logic:
# it is copied verbatim from src/c/, so Feature-A-style changes to the real
# implementation propagate to the next export automatically.

_Q14_COEFFS_TYPEDEF_RE = re.compile(r"typedef struct \{.*?\}\s*q14_coeffs_t;", re.DOTALL)
_BIQUAD_INCLUDE_LINE = '#include "filter_design.h"'


def _extract_q14_coeffs_typedef(filter_design_h_text: str) -> str:
    """Pulls the `q14_coeffs_t` typedef out of src/c/filter_design.h's text,
    so the standalone package header (below) never hand-duplicates that
    struct -- it's extracted from the real source, not retyped."""
    match = _Q14_COEFFS_TYPEDEF_RE.search(filter_design_h_text)
    if match is None:
        raise ExportError(
            "could not find the q14_coeffs_t typedef in src/c/filter_design.h -- "
            "firmware package generation is out of sync with the source (expected a "
            "'typedef struct { ... } q14_coeffs_t;' block)"
        )
    return match.group(0)


def render_standalone_biquad_header(src_dir: Path = C_SRC_DIR) -> str:
    """Renders a dependency-free variant of src/c/biquad_q14.h for the
    firmware package: its `#include "filter_design.h"` line is replaced by a
    `q14_coeffs_t` typedef extracted from the real src/c/filter_design.h,
    rather than a second hand-typed copy of that struct. This avoids a real
    naming collision -- the package also contains a *generated coefficient*
    header also named `filter_design.h` (see render_firmware_package) -- and
    means firmware doesn't pull in filter_design_lp/hp/bp/ap() (the
    host-side, ctypes-only coefficient designer), which it never needs since
    coefficients are already frozen constants.
    """
    try:
        biquad_h_text = (src_dir / "biquad_q14.h").read_text(encoding="utf-8")
        filter_design_h_text = (src_dir / "filter_design.h").read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"failed to read firmware source under {src_dir}: {exc}") from exc

    if _BIQUAD_INCLUDE_LINE not in biquad_h_text:
        raise ExportError(
            f"expected {src_dir / 'biquad_q14.h'} to contain the literal line "
            f"{_BIQUAD_INCLUDE_LINE!r} -- firmware package generation is out of sync "
            "with the source"
        )
    typedef_block = _extract_q14_coeffs_typedef(filter_design_h_text)

    standalone_block = (
        "/* Standalone firmware-package variant: q14_coeffs_t is inlined below\n"
        " * (extracted from src/c/filter_design.h at export time) instead of\n"
        " * #include-ing that header -- firmware doesn't need\n"
        " * filter_design_lp/hp/bp/ap() (coefficients are already frozen\n"
        " * constants), so this file has no dependency beyond <stdint.h>. */\n"
        f"{typedef_block}"
    )
    return biquad_h_text.replace(_BIQUAD_INCLUDE_LINE, standalone_block, 1)


def render_firmware_example(snapshot: ExportSnapshot) -> str:
    """Generates a C source demonstrating the specific chain in `snapshot`:
    one biquad_q14_state_t per active block, initialized from its FILT<n>_*
    coefficient defines, chained in series (stage n's output feeds stage
    n+1, per CONTRACTS.md §12's series-only topology) through
    `process_chain()` -- the function a real integration calls once per
    sample. `main()` is an illustrative compile-and-run demo only.
    """
    lines = [
        "/* Generated by IIR Filter Design Suite -- example integration for the",
        f" * exported design (fs = {snapshot.fs:g} Hz). process_chain() is the",
        " * function to call once per real sample in your own integration;",
        " * main() below is only a compile-and-run demo. */",
        "",
        '#include "filter_design.h"',
        '#include "biquad_q14.h"',
        "#include <stdio.h>",
        "",
    ]
    for block in snapshot.blocks:
        n = block.position
        lines.append(
            f"static const q14_coeffs_t coeffs{n} = "
            f"{{ FILT{n}_B0, FILT{n}_B1, FILT{n}_B2, FILT{n}_A1, FILT{n}_A2 }};"
        )
        lines.append(f"static biquad_q14_state_t state{n};")
    lines.append("")
    lines.append("void filter_chain_init(void) {")
    for block in snapshot.blocks:
        n = block.position
        lines.append(f"    biquad_q14_init(&state{n}, &coeffs{n});")
    lines.append("}")
    lines.append("")
    topology = " -> ".join(f"FILT{b.position}" for b in snapshot.blocks)
    lines.append(f"/* Series cascade, matching this design's chain order: {topology} */")
    lines.append("int16_t process_chain(int16_t x) {")
    lines.append("    int16_t y = x;")
    for block in snapshot.blocks:
        n = block.position
        lines.append(f"    y = biquad_q14_process(&state{n}, y);")
    lines.append("    return y;")
    lines.append("}")
    lines.append("")
    lines.append(f"/* Illustrative only: runs a {FIRMWARE_EXAMPLE_N_SAMPLES}-sample Q14 unit")
    lines.append(" * impulse through the cascade and prints each output sample, one per line. */")
    lines.append("int main(void) {")
    lines.append("    filter_chain_init();")
    lines.append(f"    for (int n = 0; n < {FIRMWARE_EXAMPLE_N_SAMPLES}; n++) {{")
    lines.append("        int16_t x = (n == 0) ? (int16_t)Q14_SCALE : 0;")
    lines.append("        int16_t y = process_chain(x);")
    lines.append('        printf("%d\\n", (int)y);')
    lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_firmware_readme(snapshot: ExportSnapshot) -> str:
    topology = " -> ".join(f"FILT{b.position} ({b.kind})" for b in snapshot.blocks)
    return (
        "# Firmware package\n"
        "\n"
        f"Generated by IIR Filter Design Suite, {snapshot.generated_at.isoformat()}.\n"
        f"fs = {snapshot.fs:g} Hz. Chain topology (series): {topology}.\n"
        "\n"
        "This folder is self-contained -- copy it into another project as-is, no\n"
        "other file from this suite is required.\n"
        "\n"
        "## Contents\n"
        "\n"
        "- `filter_design.h` -- generated Q14 coefficients for this design\n"
        "  (`FILT<n>_*` defines). Same content as the sibling top-level file.\n"
        "- `biquad_q14.h` / `biquad_q14.c` -- the Direct Form 1 Q14 biquad\n"
        "  implementation. This header variant inlines its own `q14_coeffs_t`\n"
        "  definition instead of including a separate type header, so this\n"
        "  folder has no dependency outside itself.\n"
        "- `example.c` -- generated integration example: `filter_chain_init()`\n"
        "  builds the cascade's state from the coefficients above;\n"
        "  `process_chain(x)` runs one Q14 sample through the full series\n"
        "  cascade and returns the result. `main()` is an illustrative demo\n"
        "  only (feeds a unit impulse, prints the output) -- not part of the\n"
        "  integration API.\n"
        "\n"
        "## Integration\n"
        "\n"
        "1. Copy this folder into your firmware project.\n"
        "2. Call `filter_chain_init()` once at startup.\n"
        "3. Call `process_chain(x)` once per input sample in your real-time loop.\n"
        "4. Remove or replace `example.c`'s `main()` -- it's a compile-and-run\n"
        "   demo, not part of the integration API.\n"
    )


def render_firmware_package(snapshot: ExportSnapshot, output_dir: Path, *, src_dir: Path = C_SRC_DIR) -> Path:
    """Writes the `firmware/` subfolder (see module docstring) under
    `output_dir` and returns its path."""
    firmware_dir = output_dir / "firmware"
    try:
        firmware_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ExportError(f"failed to create firmware package directory {firmware_dir}: {exc}") from exc

    _write_text_lf(firmware_dir / "filter_design.h", render_header(snapshot))
    _write_text_lf(firmware_dir / "biquad_q14.h", render_standalone_biquad_header(src_dir), encoding="utf-8")
    try:
        biquad_c_text = (src_dir / "biquad_q14.c").read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"failed to read {src_dir / 'biquad_q14.c'}: {exc}") from exc
    _write_text_lf(firmware_dir / "biquad_q14.c", biquad_c_text, encoding="utf-8")
    _write_text_lf(firmware_dir / "example.c", render_firmware_example(snapshot), encoding="utf-8")
    _write_text_lf(firmware_dir / "README.md", render_firmware_readme(snapshot), encoding="utf-8")

    return firmware_dir


# -- PNG plots -----------------------------------------------------------------


def _save_bode_png(path: Path, ideal, q14, title: str) -> None:
    fig = Figure(figsize=(6.0, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax_mag, ax_phase = fig.subplots(2, 1, sharex=True)

    ax_mag.set_xscale("log")
    ax_mag.plot(ideal.freq_hz, ideal.magnitude_db, color=IDEAL_COLOR, label="Ideal")
    ax_mag.plot(q14.freq_hz, q14.magnitude_db, color=Q14_COLOR, linestyle="--", label="Q14")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title(title)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="best", fontsize="small")
    _, mag_top = ax_mag.get_ylim()
    ax_mag.set_ylim(MAGNITUDE_YLIM_FLOOR, max(mag_top, MAGNITUDE_YLIM_FLOOR))

    ax_phase.set_xscale("log")
    ax_phase.plot(ideal.freq_hz, ideal.phase_deg, color=IDEAL_COLOR, label="Ideal")
    ax_phase.plot(q14.freq_hz, q14.phase_deg, color=Q14_COLOR, linestyle="--", label="Q14")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.set_ylim(*PHASE_YLIM)

    _save_figure(fig, path)


def _coefficient_sweep_curve(
    fs: float, design_at: Callable[[float], FilterDesign], backend: NativeBackend, n: int = ERROR_SWEEP_PLOT_N
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point max-abs coefficient error across the sweep grid, for plotting.

    Mirrors `error_analysis.coefficient_sweep()`'s grid and per-point error
    computation (same `[COEFFICIENT_SWEEP_FLOOR_HZ, fc_max(fs)]` domain, same
    ideal-vs-dequantized-Q14 comparison) so the plotted curve is consistent
    with that function's summary stats -- but at a smaller, plot-appropriate
    point count. No filter-design or quantization math is reimplemented
    here: every point still goes through `FilterDesign.ideal_coefficients()`
    / `.q14_coefficients(backend)`, exactly like `coefficient_sweep()` does.
    """
    hi = fc_max(fs)
    grid = np.linspace(COEFFICIENT_SWEEP_FLOOR_HZ, hi, n)
    errs = np.empty(n)
    for i, x in enumerate(grid):
        filt = design_at(float(x))
        ideal = filt.ideal_coefficients()
        q14_float = filt.q14_coefficients(backend).to_float()
        errs[i] = max(abs(getattr(ideal, name) - getattr(q14_float, name)) for name in COEFFICIENT_NAMES)
    return grid, errs


def _save_error_sweep_png(path: Path, freq_hz: np.ndarray, err: np.ndarray, title: str) -> None:
    fig = Figure(figsize=(6.0, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.plot(freq_hz, err, color=Q14_COLOR)
    ax.set_xlabel("Sweep frequency (Hz)")
    ax.set_ylabel("Max |coefficient error|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, path)


def _save_figure(fig: Figure, path: Path) -> None:
    try:
        fig.savefig(str(path), format="png", dpi=150)
    except OSError as exc:
        raise ExportError(f"failed to write PNG {path}: {exc}") from exc


# -- PDF report (CONCEPT.md §7) ------------------------------------------------


def render_pdf(
    snapshot: ExportSnapshot,
    path: Path,
    combined_bode_png: Path,
    block_bode_pngs: Mapping[int, Path],
) -> None:
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle("Code", parent=styles["Normal"], fontName="Courier", fontSize=6.5, leading=8)
    story = []

    # 1. Design summary
    story.append(Paragraph("IIR Filter Design Suite - Export Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {snapshot.generated_at.isoformat()}", styles["Normal"]))
    story.append(Paragraph(f"fs = {snapshot.fs:g} Hz", styles["Normal"]))
    topology = " -&gt; ".join(f"{b.name} ({b.kind})" for b in snapshot.blocks)
    story.append(Paragraph(f"Chain topology (series): {topology}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # 2. Per-filter sections -- each starts on its own page (a PageBreak
    # before every block, including the first, keeps the summary above alone
    # on page 1); this is a page-count floor, not a one-page-per-filter cap
    # -- a block with a long coefficient-sweep note can still spill onto the
    # next page.
    for block in snapshot.blocks:
        story.append(PageBreak())
        story.append(Paragraph(f"{block.name}: {FILTER_KIND_NAMES[block.kind]}", styles["Heading2"]))
        story.append(Paragraph(_filter_description(block), styles["Normal"]))

        png = block_bode_pngs.get(block.position)
        if png is not None and png.is_file():
            story.append(Image(str(png), width=5.0 * inch, height=3.75 * inch))

        table_data = [["coefficient", "ideal (float64)", "Q14 (int16)", "Q14 (float)"]]
        for coef in COEFFICIENT_NAMES:
            ideal_v = getattr(block.ideal_coefficients, coef)
            q14_v = getattr(block.q14_coefficients, coef)
            q14f_v = getattr(block.q14_coefficients.to_float(), coef)
            table_data.append([coef, f"{ideal_v:.8f}", str(q14_v), f"{q14f_v:.8f}"])
        story.append(Table(table_data, style=_TABLE_STYLE))

        story.append(
            Paragraph(
                f"Response error vs Q14 -- max: {block.response_error.max_db:.4f} dB, "
                f"RMS: {block.response_error.rms_db:.4f} dB -- "
                f"<b>{'PASS' if block.response_passed else 'FAIL'}</b> "
                f"(threshold {RESPONSE_PASS_THRESHOLD_DB} dB)",
                styles["Normal"],
            )
        )
        if block.coefficient_sweep is not None:
            sweep = block.coefficient_sweep
            story.append(
                Paragraph(
                    f"Coefficient sweep -- max abs: {sweep.max_abs:.3e}, RMS abs: {sweep.rms_abs:.3e}, "
                    f"worst at {sweep.worst_frequency_hz:.1f} Hz (diagnostic, no pass/fail threshold)",
                    styles["Normal"],
                )
            )
        else:
            story.append(Paragraph(block.sweep_unavailable_reason or "Coefficient sweep unavailable.", styles["Normal"]))
        story.append(Spacer(1, 12))

    # 3. Combined chain -- own page, same as every per-filter section above
    # (a block with a long coefficient-sweep note can still spill onto the
    # next page; this is a page-count floor, not a one-page cap).
    story.append(PageBreak())
    story.append(Paragraph("Combined chain", styles["Heading2"]))
    if combined_bode_png.is_file():
        story.append(Image(str(combined_bode_png), width=5.0 * inch, height=3.75 * inch))
    story.append(
        Paragraph(
            f"Combined response error vs Q14 -- max: {snapshot.combined_response_error.max_db:.4f} dB, "
            f"RMS: {snapshot.combined_response_error.rms_db:.4f} dB -- "
            f"<b>{'PASS' if snapshot.combined_passed else 'FAIL'}</b> (threshold {RESPONSE_PASS_THRESHOLD_DB} dB)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    # 4. C header listing
    story.append(Paragraph("C header listing (filter_design.h)", styles["Heading2"]))
    for line in render_header(snapshot).splitlines():
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        story.append(Paragraph(escaped or "&nbsp;", code_style))
    story.append(Spacer(1, 12))

    # 5. Test results
    story.append(Paragraph("Test results", styles["Heading2"]))
    result_data = [["Block", "Kind", "Max error (dB)", "Result"]]
    for block in snapshot.blocks:
        result_data.append(
            [block.name, block.kind, f"{block.response_error.max_db:.4f}", "PASS" if block.response_passed else "FAIL"]
        )
    result_data.append(
        ["Combined", "-", f"{snapshot.combined_response_error.max_db:.4f}", "PASS" if snapshot.combined_passed else "FAIL"]
    )
    story.append(Table(result_data, style=_TABLE_STYLE))

    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    try:
        doc.build(story)
    except OSError as exc:
        raise ExportError(f"failed to write PDF {path}: {exc}") from exc


_TABLE_STYLE = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.5, "#999999"),
        ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]
)


# -- Export directory + top-level orchestration --------------------------------


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    snapshot: ExportSnapshot
    header_path: Path
    pdf_path: Path
    combined_bode_png: Path
    block_bode_pngs: Mapping[int, Path]
    error_sweep_pngs: Mapping[int, Path]
    firmware_dir: Path


def _make_export_dir(output_root: Path, generated_at: datetime) -> Path:
    """Creates a fresh `export_YYYYMMDD_HHMMSS[_N]` directory under `output_root`.

    The `_N` suffix only engages on a same-second collision (e.g. two
    exports fired within one wall-clock second); it keeps directory
    creation from failing outright rather than silently overwriting a
    prior export.
    """
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"failed to create export root {output_root}: {exc}") from exc

    base_name = f"export_{generated_at.strftime('%Y%m%d_%H%M%S')}"
    for suffix in range(1000):
        candidate = output_root / (base_name if suffix == 0 else f"{base_name}_{suffix}")
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise ExportError(f"failed to create export directory {candidate}: {exc}") from exc
    raise ExportError(f"could not create a unique export directory under {output_root} (too many collisions)")


def export_design(
    chain: FilterChain,
    backend: NativeBackend,
    output_root: Path | str,
    *,
    now: datetime | None = None,
) -> ExportResult:
    """Forces a fresh validation pass and writes the full export file set.

    Raises `ValueError` for an empty chain or any invalid block (before any
    filesystem I/O happens), and `ExportError` for filesystem/plotting/PDF
    failures encountered while writing the file set.
    """
    snapshot = build_snapshot(chain, backend, now=now)
    output_root = Path(output_root)
    export_dir = _make_export_dir(output_root, snapshot.generated_at)

    freq = bode_grid(snapshot.fs)

    combined_ideal = chain.combined_ideal_response(freq)
    combined_q14 = chain.combined_q14_response(freq, backend)
    combined_png = export_dir / "bode_combined.png"
    _save_bode_png(combined_png, combined_ideal, combined_q14, "Combined chain")

    block_pngs: dict[int, Path] = {}
    sweep_pngs: dict[int, Path] = {}
    for block, block_snap in zip(_active_blocks(chain), snapshot.blocks):
        filt = block.filter
        ideal_resp = filt.ideal_response(freq)
        q14_resp = filt.q14_response(freq, backend)
        png_path = export_dir / f"bode_{block.kind.lower()}_{block_snap.position}.png"
        _save_bode_png(png_path, ideal_resp, q14_resp, f"{block_snap.name}: {FILTER_KIND_NAMES[block.kind]}")
        block_pngs[block_snap.position] = png_path

        design_at = _sweep_design_at(block, chain.fs)
        if design_at is not None:
            sweep_path = export_dir / f"error_sweep_{block_snap.position}.png"
            curve_freq, curve_err = _coefficient_sweep_curve(chain.fs, design_at, backend)
            _save_error_sweep_png(
                sweep_path, curve_freq, curve_err, f"{block_snap.name}: {block.kind} coefficient error sweep"
            )
            sweep_pngs[block_snap.position] = sweep_path

    header_path = export_dir / "filter_design.h"
    _write_header(header_path, snapshot)

    firmware_dir = render_firmware_package(snapshot, export_dir)

    pdf_path = export_dir / "report.pdf"
    render_pdf(snapshot, pdf_path, combined_png, block_pngs)

    return ExportResult(
        output_dir=export_dir,
        snapshot=snapshot,
        header_path=header_path,
        pdf_path=pdf_path,
        combined_bode_png=combined_png,
        block_bode_pngs=MappingProxyType(block_pngs),
        error_sweep_pngs=MappingProxyType(sweep_pngs),
        firmware_dir=firmware_dir,
    )
