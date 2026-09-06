"""ctypes ABI, compiler discovery, and actionable-failure tests for c_codegen.py.

Covers CONTRACTS.md §8 (struct layout, function signatures, error codes,
nonzero-rc -> RuntimeError) and §9 (compiler discovery order, actionable
compile-failure reporting).
"""

from __future__ import annotations

import ctypes

import pytest

from c_codegen import (
    CompilerNotFoundError,
    NativeCallError,
    NativeCompileError,
    Q14Coeffs,
    compile_shared_library,
    discover_compiler,
)


def test_q14coeffs_struct_layout_matches_contract():
    fields = Q14Coeffs._fields_
    names = [name for name, _ in fields]
    assert names == ["b0", "b1", "b2", "a1", "a2"]
    assert all(ctype is ctypes.c_int16 for _, ctype in fields)


def test_q14coeffs_and_biquadstate_are_16bit_only_no_padding():
    """CONTRACTS.md §7/§8: all-int16_t layout, no 64-bit type anywhere in the
    native ABI. A cheap struct-size guard against a future field-width or
    padding regression (int16-only fields need no alignment padding, unlike a
    mixed int16/int32 layout would)."""
    from c_codegen import BiquadState

    assert ctypes.sizeof(Q14Coeffs) == 10  # 5 x int16
    assert ctypes.sizeof(BiquadState) == 18  # Q14Coeffs (10) + 4 x int16


def test_discover_compiler_finds_something_on_this_machine():
    # This dev/CI machine has gcc installed; exercised for real, no mocking.
    compiler = discover_compiler()
    assert compiler


def test_discover_compiler_honors_cc_env_var(monkeypatch):
    monkeypatch.setenv("CC", "totally-not-a-real-compiler")
    assert discover_compiler() == "totally-not-a-real-compiler"


def test_discover_compiler_raises_actionable_error_when_nothing_found(monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr("c_codegen.shutil.which", lambda name: None)
    with pytest.raises(CompilerNotFoundError) as exc_info:
        discover_compiler()
    message = str(exc_info.value)
    assert "gcc" in message and "clang" in message and "CC" in message


def test_compile_with_nonexistent_compiler_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CC", "totally-not-a-real-compiler")
    with pytest.raises(CompilerNotFoundError) as exc_info:
        compile_shared_library(tmp_path)
    assert "totally-not-a-real-compiler" in str(exc_info.value)


def test_compile_missing_source_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr("c_codegen.C_SRC_DIR", tmp_path / "nowhere")
    with pytest.raises(NativeCompileError) as exc_info:
        compile_shared_library(tmp_path)
    assert "Missing C source file" in str(exc_info.value)


def test_compile_bad_source_reports_compiler_stderr(monkeypatch, tmp_path):
    bad_src_dir = tmp_path / "badsrc"
    bad_src_dir.mkdir()
    (bad_src_dir / "filter_design.c").write_text("this is not valid C {{{\n")
    (bad_src_dir / "biquad_q14.c").write_text("this is not valid C either }}}\n")
    monkeypatch.setattr("c_codegen.C_SRC_DIR", bad_src_dir)
    with pytest.raises(NativeCompileError) as exc_info:
        compile_shared_library(tmp_path)
    message = str(exc_info.value)
    assert "Compilation failed" in message
    assert "stderr" in message.lower()


def test_compile_shared_library_produces_loadable_library(tmp_path):
    lib_path = compile_shared_library(tmp_path)
    assert lib_path.is_file()
    lib = ctypes.CDLL(str(lib_path))
    assert hasattr(lib, "filter_design_lp")
    assert hasattr(lib, "biquad_q14_process")


def test_design_functions_return_zero_on_success(native_backend):
    coeffs = native_backend.design_lp(3000.0, 13333.0)
    assert isinstance(coeffs.b0, int)


@pytest.mark.parametrize(
    ("call", "expected_rc"),
    [
        (lambda b: b.design_lp(3000.0, -1.0), -2),  # fs <= 0
        (lambda b: b.design_lp(-100.0, 13333.0), -1),  # freq out of (0, fs/2)
        (lambda b: b.design_lp(7000.0, 13333.0), -1),  # freq >= fs/2
        (lambda b: b.design_bp(4000.0, 2000.0, 13333.0), -3),  # f_low >= f_high
        (lambda b: b.design_ap(3000.0, 13333.0, -1.0), -4),  # q out of (0, inf)
        (lambda b: b.design_ap(3000.0, 13333.0, 0.0), -4),
    ],
)
def test_invalid_input_raises_native_call_error_with_correct_code(native_backend, call, expected_rc):
    with pytest.raises(NativeCallError) as exc_info:
        call(native_backend)
    assert str(expected_rc) in str(exc_info.value)
