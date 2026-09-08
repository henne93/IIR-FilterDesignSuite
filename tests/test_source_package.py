"""Standalone-compile proof for the generated source/ package (CONTRACTS.md §10).

Distinct from test_firmware_harness.py, which proves the *generated
coefficient* header combines correctly with this suite's own
src/c/biquad_q14.* (both are pulled in via -I flags, i.e. never actually
detached from the repo). This file proves the stronger claim made for the
source/ package as a whole: it is genuinely self-contained -- copy it
anywhere, no reference to src/c/ at all, and it still compiles, links, and
runs, with the same numerical behavior as the ctypes-backed NativeBackend
used everywhere else in this suite.

Also distinct from export.py's own `run_c_validation()`, which runs this
same kind of compile+run+cross-check as a *production* step against each
export's own freshly-written (non-detached) source/ files -- see its
docstring for why it is implemented in-process there rather than shelling
out to pytest at runtime.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import astuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

from export import BIQUAD_DIRNAME, export_design
from filters import FilterChain

FS = 13333.0
FIXED_NOW = datetime(2026, 8, 18, 10, 30, 0, tzinfo=timezone(timedelta(hours=2)))

# One literal per design-function argument, formatted via repr() when embedded
# in the harness source below so the C double literal round-trips to the exact
# same bit pattern ctypes hands the NativeBackend -- both then narrow it to
# `float` (Q14 design functions take float args) via the same IEEE conversion,
# so the two paths compute from bit-identical inputs.
_DESIGN_FC = 1000.0
_DESIGN_F_LOW = 500.0
_DESIGN_F_HIGH = 2000.0
_DESIGN_Q = 0.7071
_DESIGN_GAIN_DB = 6.0

_DESIGN_HARNESS_C = """\
#include <stdio.h>
#include "filter_design_calc.h"

static void print_coeffs(const q14_coeffs_t *c) {{
    printf("%d %d %d %d %d\\n", c->b0, c->b1, c->b2, c->a1, c->a2);
}}

int main(void) {{
    q14_coeffs_t c;
    filter_design_lp({fc!r}, {fs!r}, &c); print_coeffs(&c);
    filter_design_hp({fc!r}, {fs!r}, &c); print_coeffs(&c);
    filter_design_bp({f_low!r}, {f_high!r}, {fs!r}, &c); print_coeffs(&c);
    filter_design_ap({fc!r}, {fs!r}, {q!r}, &c); print_coeffs(&c);
    filter_design_pk({fc!r}, {fs!r}, {q!r}, {gain_db!r}, &c); print_coeffs(&c);
    return 0;
}}
""".format(
    fc=_DESIGN_FC,
    fs=FS,
    f_low=_DESIGN_F_LOW,
    f_high=_DESIGN_F_HIGH,
    q=_DESIGN_Q,
    gain_db=_DESIGN_GAIN_DB,
)

# Canonical compile command (export.py module docstring / source/README.md):
#   gcc -Wall -Wextra -Werror -std=c11 -I biquad_q14/inc app_template/example.c
#       biquad_q14/src/biquad_q14.c biquad_q14/src/filter_design_calc.c -o demo -lm


def _build_and_run_detached_copy(tmp_path: Path, source_dir: Path) -> list[int]:
    """Copies source_dir to a location with no relationship to the repo,
    compiles it there with -I pointed only at that copy, and runs it --
    the actual "drop this folder into a project elsewhere" scenario."""
    detached = tmp_path / "detached_elsewhere" / "source"
    shutil.copytree(source_dir, detached)
    biquad_dir = detached / BIQUAD_DIRNAME
    app_template_dir = detached / "app_template"

    binary = detached / "demo"
    compile_cmd = [
        "gcc", "-Wall", "-Wextra", "-Werror", "-std=c11",
        "-I", str(biquad_dir / "inc"),  # the ONLY include path -- no src/c/ anywhere
        str(app_template_dir / "example.c"), str(biquad_dir / "src" / "biquad_q14.c"),
        str(biquad_dir / "src" / "filter_design_calc.c"),
        "-o", str(binary), "-lm",
    ]
    compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr
    return [int(line) for line in run_proc.stdout.split()]


def test_generated_source_package_compiles_links_and_runs_standalone(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    chain.add_block("HP", fc=1000.0)
    result = export_design(chain, native_backend, tmp_path / "export_root", now=FIXED_NOW)

    samples = _build_and_run_detached_copy(tmp_path, result.source_dir)

    from export import FIRMWARE_EXAMPLE_N_SAMPLES

    assert len(samples) == FIRMWARE_EXAMPLE_N_SAMPLES

    # Bit-exact cross-check against a Python-computed series cascade: since
    # each stage is a stateful-but-parameter-fixed LTI filter, chaining whole
    # sample sequences through process_samples() one stage at a time gives
    # identical results to interleaved per-sample chaining (what example.c's
    # process_chain() actually does).
    impulse = [16384] + [0] * (FIRMWARE_EXAMPLE_N_SAMPLES - 1)
    stage_input = impulse
    for block_snap in result.snapshot.blocks:
        stage_input = native_backend.process_samples(block_snap.q14_coefficients, stage_input)
    assert samples == stage_input


def test_generated_source_package_design_functions_match_native_backend(tmp_path, native_backend):
    """Proves the bundled filter_design_calc.{h,c} (CONTRACTS.md §10's
    reversal of the earlier "firmware doesn't need filter_design_lp/hp/bp/ap()"
    decision) actually work, standalone, detached from src/c/ -- not just that
    they compile as inert bystanders in the package. A harness calling
    filter_design_lp/hp/bp/ap/pk() directly, compiled only against the
    detached copy, must produce the same Q14 coefficients as the ctypes
    NativeBackend built from the same (unrenamed) source -- which is itself
    cross-checked against the Python/scipy "ideal" coefficients across the
    full parameter domain in tests/test_native_coefficients.py. A divergence
    here would mean the package's renamed copy silently drifted from the real
    design math, e.g. from a bad #include rewrite picking up stale values.
    """
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    result = export_design(chain, native_backend, tmp_path / "export_root", now=FIXED_NOW)

    detached = tmp_path / "detached_design_calc" / "source"
    shutil.copytree(result.source_dir, detached)
    biquad_dir = detached / BIQUAD_DIRNAME

    harness_src = detached / "design_harness.c"
    harness_src.write_text(_DESIGN_HARNESS_C)
    binary = detached / "design_harness"
    compile_cmd = [
        "gcc", "-Wall", "-Wextra", "-Werror", "-std=c11",
        "-I", str(biquad_dir / "inc"),  # the ONLY include path -- no src/c/ anywhere
        str(harness_src), str(biquad_dir / "src" / "filter_design_calc.c"),
        "-o", str(binary), "-lm",
    ]
    compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr
    lines = run_proc.stdout.strip().splitlines()
    assert len(lines) == 5
    harness_lp, harness_hp, harness_bp, harness_ap, harness_pk = (tuple(int(v) for v in line.split()) for line in lines)

    assert harness_lp == astuple(native_backend.design_lp(_DESIGN_FC, FS))
    assert harness_hp == astuple(native_backend.design_hp(_DESIGN_FC, FS))
    assert harness_bp == astuple(native_backend.design_bp(_DESIGN_F_LOW, _DESIGN_F_HIGH, FS))
    assert harness_ap == astuple(native_backend.design_ap(_DESIGN_FC, FS, _DESIGN_Q))
    assert harness_pk == astuple(native_backend.design_pk(_DESIGN_FC, FS, _DESIGN_Q, _DESIGN_GAIN_DB))


def test_generated_source_package_is_stateful_not_a_passthrough(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    result = export_design(chain, native_backend, tmp_path / "export_root", now=FIXED_NOW)

    samples = _build_and_run_detached_copy(tmp_path, result.source_dir)

    assert any(s != 0 for s in samples[1:]), "expected a non-trivial IIR tail after the impulse"
