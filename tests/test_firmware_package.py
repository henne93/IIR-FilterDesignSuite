"""Standalone-compile proof for the generated firmware/ package (CONTRACTS.md §10).

Distinct from test_firmware_harness.py, which proves the *top-level*
generated header combines correctly with this suite's own src/c/biquad_q14.*
(both are pulled in via -I flags, i.e. never actually detached from the
repo). This file proves the stronger claim made for the firmware/ subfolder
specifically: it is genuinely self-contained -- copy it anywhere, no
reference to src/c/ at all, and it still compiles, links, and runs, with the
same numerical behavior as the ctypes-backed NativeBackend used everywhere
else in this suite.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from export import export_design
from filters import FilterChain

FS = 13333.0
FIXED_NOW = datetime(2026, 8, 18, 10, 30, 0, tzinfo=timezone(timedelta(hours=2)))


def _build_and_run_detached_copy(tmp_path: Path, firmware_dir: Path) -> list[int]:
    """Copies firmware_dir to a location with no relationship to the repo,
    compiles it there with -I pointed only at that copy, and runs it --
    the actual "drop this folder into a project elsewhere" scenario."""
    detached = tmp_path / "detached_elsewhere" / "firmware"
    shutil.copytree(firmware_dir, detached)

    binary = detached / "demo"
    compile_cmd = [
        "gcc", "-Wall", "-Wextra", "-Werror", "-std=c11",
        "-I", str(detached),  # the ONLY include path -- no src/c/ anywhere
        str(detached / "example.c"), str(detached / "biquad_q14.c"),
        "-o", str(binary), "-lm",
    ]
    compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr
    return [int(line) for line in run_proc.stdout.split()]


def test_generated_firmware_package_compiles_links_and_runs_standalone(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    chain.add_block("HP", fc=1000.0)
    result = export_design(chain, native_backend, tmp_path / "export_root", now=FIXED_NOW)

    samples = _build_and_run_detached_copy(tmp_path, result.firmware_dir)

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


def test_generated_firmware_package_is_stateful_not_a_passthrough(tmp_path, native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    result = export_design(chain, native_backend, tmp_path / "export_root", now=FIXED_NOW)

    samples = _build_and_run_detached_copy(tmp_path, result.firmware_dir)

    assert any(s != 0 for s in samples[1:]), "expected a non-trivial IIR tail after the impulse"
