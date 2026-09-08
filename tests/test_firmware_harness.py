"""Generated-header firmware harness test (CONTRACTS.md §7, §10, §13).

`filter_design.h` (produced by `export.render_header()`, written to
`source/biquad_q14/gen/filter_design.h`) is coefficient-only -- plain
`#define FILT<n>_*` integer literals plus `Q14_SCALE`/`Q14_TO_FLOAT`
(CONTRACTS.md §10). It does not define `q14_coeffs_t`, the biquad state
struct, or `process()` -- a firmware integrator combines it with
`biquad_q14.{h,c}` themselves, exactly as this test does. (The exported
`source/` package *also* bundles a generated, standalone copy of
`biquad_q14.{h,c}` alongside the generated header -- see
`tests/test_source_package.py` -- but this test is only about the
*generated coefficient header* combined with this suite's own `src/c/`
reference source, distinct from that fully-bundled package.)

Two other tests already cover the pieces on either side of that seam:
  - `test_export.py::test_generated_header_compiles_with_gcc` proves the
    generated header alone is standalone-compilable (no undefined `Q14(...)`
    macro, `#pragma once` guard is idempotent).
  - `test_native_impulse.py` proves `biquad_q14_process()` itself is
    numerically correct, via ctypes, against a `scipy.signal.dimpulse()`
    reference.

Neither proves the *combination* a firmware target actually ships: this
file compiles a small harness that `#include`s the generated header
together with the app's own `biquad_q14.h`, links against `biquad_q14.c`,
and *runs* the resulting binary with GCC -- then cross-checks its actual
process() output against the same `NativeBackend.process_impulse()` ctypes
call used elsewhere in this suite. A divergence here would mean the
generated header's values don't drive `biquad_q14.c` the same way the app's
own ctypes path does -- something neither test above could catch alone.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from export import export_design
from filters import FilterChain

FS = 13333.0
FIXED_NOW = datetime(2026, 8, 18, 10, 30, 0, tzinfo=timezone(timedelta(hours=2)))
C_SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "c"
N_SAMPLES = 8

# Quoted #include resolution (both GCC and Clang) checks the including
# file's own directory before the -I search path, so:
#   - harness.c's own "filter_design.h" resolves to the *generated* header
#     (harness.c has no local filter_design.h; the compile command passes
#     the generated header's directory first on -I, per compiler behavior
#     immediately below).
#   - biquad_q14.h's internal "filter_design.h" (src/c/biquad_q14.h) always
#     resolves to src/c/filter_design.h (same directory as biquad_q14.h
#     itself), which is what actually defines q14_coeffs_t -- regardless of
#     -I order. Both physical files use `#pragma once`, but by distinct
#     path, so both get fully included exactly once with no conflict; the
#     one macro both define identically (Q14_SCALE 16384) is a
#     standard-permitted benign redefinition, not a -Werror hit.
_HARNESS_C = """\
#include <stdio.h>
#include "filter_design.h"
#include "biquad_q14.h"

int main(void) {
    q14_coeffs_t c = { FILT1_B0, FILT1_B1, FILT1_B2, FILT1_A1, FILT1_A2 };
    biquad_q14_state_t state;
    biquad_q14_init(&state, &c);

    for (int n = 0; n < %d; n++) {
        int32_t x = (n == 0) ? Q14_SCALE : 0;
        int32_t y = biquad_q14_process(&state, x);
        printf("%%d\\n", (int)y);
    }
    return 0;
}
""" % N_SAMPLES


def test_firmware_harness_compiles_links_and_matches_ctypes_reference(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)
    q14_coeffs = result.snapshot.blocks[0].q14_coefficients

    harness_src = tmp_path / "harness.c"
    harness_src.write_text(_HARNESS_C)
    harness_bin = tmp_path / "harness"

    compile_cmd = [
        "gcc", "-Wall", "-Wextra", "-Werror", "-std=c11",
        "-I", str(result.header_path.parent),  # generated filter_design.h (coefficient-only)
        "-I", str(C_SRC_DIR),          # biquad_q14.h/.c reference firmware source
        str(harness_src), str(C_SRC_DIR / "biquad_q14.c"),
        "-o", str(harness_bin), "-lm",
    ]
    compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run([str(harness_bin)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr
    harness_samples = [int(line) for line in run_proc.stdout.split()]
    assert len(harness_samples) == N_SAMPLES

    reference_samples = native_backend.process_impulse(q14_coeffs, N_SAMPLES)
    assert harness_samples == reference_samples


def test_firmware_harness_is_stateful_not_a_passthrough(tmp_path, native_backend):
    # Sanity check independent of the ctypes cross-check above: a biquad's
    # impulse response has a non-trivial IIR tail, so an all-zero (or
    # all-impulse-only) harness output would indicate the harness silently
    # isn't driving process()'s real state-update path.
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    result = export_design(chain, native_backend, tmp_path, now=FIXED_NOW)

    harness_src = tmp_path / "harness.c"
    harness_src.write_text(_HARNESS_C)
    harness_bin = tmp_path / "harness"
    compile_cmd = [
        "gcc", "-Wall", "-Wextra", "-Werror", "-std=c11",
        "-I", str(result.header_path.parent), "-I", str(C_SRC_DIR),
        str(harness_src), str(C_SRC_DIR / "biquad_q14.c"),
        "-o", str(harness_bin), "-lm",
    ]
    subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30, check=True)

    run_proc = subprocess.run([str(harness_bin)], capture_output=True, text=True, timeout=10, check=True)
    samples = [int(line) for line in run_proc.stdout.split()]
    assert any(s != 0 for s in samples[1:]), "expected a non-trivial IIR tail after the impulse"
