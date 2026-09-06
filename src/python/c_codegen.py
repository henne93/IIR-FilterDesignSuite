"""ctypes ABI to the compiled Q14 filter-design C library.

Implements CONTRACTS.md §8 (function signatures / ctypes ABI) and §9
(compiler discovery, compile command, process-local recompile-every-launch
cache behavior, actionable failure handling).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from filters.base import Q14Coefficients

C_SRC_DIR = Path(__file__).resolve().parent.parent / "c"
C_SOURCES = ["filter_design.c", "biquad_q14.c"]

# CONTRACTS.md §8: 0 success, -1 frequency out of (0, fs/2), -2 fs <= 0,
# -3 f_low >= f_high (BP only), -4 q out of (0, inf) (AP only).
_ERROR_MESSAGES = {
    -1: "frequency out of (0, fs/2)",
    -2: "fs <= 0",
    -3: "f_low >= f_high",
    -4: "q out of (0, inf)",
}


class CompilerNotFoundError(RuntimeError):
    """No GCC/Clang-compatible compiler could be found (CONTRACTS.md §9)."""


class NativeCompileError(RuntimeError):
    """The C sources failed to compile into a shared library."""


class NativeCallError(RuntimeError):
    """A compiled C function returned a nonzero (error) status code."""


def discover_compiler() -> str:
    """Compiler-discovery order per CONTRACTS.md §9: $CC -> gcc -> cc -> clang.

    An explicit $CC is trusted as-is (the user asked for it by name/path);
    gcc/cc/clang are only used if actually found on PATH.
    """
    cc_env = os.environ.get("CC")
    if cc_env:
        return cc_env
    for candidate in ("gcc", "cc", "clang"):
        if shutil.which(candidate):
            return candidate
    raise CompilerNotFoundError(
        "No C compiler found. The native Q14 backend requires a GCC- or "
        "Clang-compatible compiler on PATH (gcc, cc, or clang), or the $CC "
        "environment variable pointing at one. Install one (e.g. `apt "
        "install gcc` on Debian/Ubuntu, or a MinGW-w64 GCC on Windows) and "
        "try again."
    )


def _library_filename() -> str:
    return "filter_design.dll" if os.name == "nt" else "filter_design.so"


def compile_shared_library(build_dir: Path) -> Path:
    """Compiles src/c/*.c into a fresh shared library under build_dir.

    Unconditional recompile every call, per CONTRACTS.md §9 ("recompiled
    every process launch... not cached to disk across separate app
    launches") -- callers own the "reused only for that process's
    lifetime" part by calling this once and keeping the resulting library
    loaded (see NativeBackend below).
    """
    compiler = discover_compiler()

    sources = [C_SRC_DIR / name for name in C_SOURCES]
    missing = [str(src) for src in sources if not src.is_file()]
    if missing:
        raise NativeCompileError(f"Missing C source file(s): {', '.join(missing)}")

    lib_path = build_dir / _library_filename()
    cmd = [compiler, "-O2"]
    if os.name != "nt":
        cmd.append("-fPIC")
    cmd += ["-shared", "-o", str(lib_path), *(str(s) for s in sources), "-I", str(C_SRC_DIR), "-lm"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise CompilerNotFoundError(
            f"Compiler '{compiler}' could not be executed ({exc}). Install a "
            "GCC- or Clang-compatible compiler, or set $CC to a valid one."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise NativeCompileError(f"Compilation timed out after 60s: {' '.join(cmd)}") from exc

    if result.returncode != 0:
        stderr_excerpt = "\n".join(result.stderr.splitlines()[:20])
        raise NativeCompileError(
            f"Compilation failed (compiler={compiler!r}, exit={result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Compiler stderr (first 20 lines):\n{stderr_excerpt}"
        )
    if not lib_path.is_file():
        raise NativeCompileError(
            f"Compiler exited 0 but produced no library at {lib_path}. "
            f"Command: {' '.join(cmd)}"
        )
    return lib_path


class Q14Coeffs(ctypes.Structure):
    """CONTRACTS.md §8 ctypes ABI struct, matching q14_coeffs_t exactly."""

    _fields_ = [
        ("b0", ctypes.c_int32),
        ("b1", ctypes.c_int32),
        ("b2", ctypes.c_int32),
        ("a1", ctypes.c_int32),
        ("a2", ctypes.c_int32),
    ]


class BiquadState(ctypes.Structure):
    """Mirrors biquad_q14_state_t (src/c/biquad_q14.h) for the process() test hook."""

    _fields_ = [
        ("coeffs", Q14Coeffs),
        ("x1", ctypes.c_int32),
        ("x2", ctypes.c_int32),
        ("y1", ctypes.c_int32),
        ("y2", ctypes.c_int32),
    ]


def _configure_argtypes(lib: ctypes.CDLL) -> None:
    c_float = ctypes.c_float
    p_coeffs = ctypes.POINTER(Q14Coeffs)

    lib.filter_design_lp.argtypes = [c_float, c_float, p_coeffs]
    lib.filter_design_lp.restype = ctypes.c_int
    lib.filter_design_hp.argtypes = [c_float, c_float, p_coeffs]
    lib.filter_design_hp.restype = ctypes.c_int
    lib.filter_design_bp.argtypes = [c_float, c_float, c_float, p_coeffs]
    lib.filter_design_bp.restype = ctypes.c_int
    lib.filter_design_ap.argtypes = [c_float, c_float, c_float, p_coeffs]
    lib.filter_design_ap.restype = ctypes.c_int

    lib.biquad_q14_init.argtypes = [ctypes.POINTER(BiquadState), p_coeffs]
    lib.biquad_q14_init.restype = None
    lib.biquad_q14_process.argtypes = [ctypes.POINTER(BiquadState), ctypes.c_int32]
    lib.biquad_q14_process.restype = ctypes.c_int32


def _raise_for_rc(rc: int, fn_name: str) -> None:
    if rc != 0:
        reason = _ERROR_MESSAGES.get(rc, f"unknown error code {rc}")
        raise NativeCallError(
            f"{fn_name} returned error code {rc} ({reason}). Per CONTRACTS.md "
            "§8 this should be unreachable in normal operation -- Python is "
            "expected to validate parameters before calling into C (§5); "
            "treat this as a bug in that validation, not a normal error path."
        )


class NativeBackend:
    """ctypes-backed NativeBackend implementing filters.base.NativeBackend.

    Compiles a fresh, process-local shared library (CONTRACTS.md §9) into a
    private temp directory owned by this instance and keeps it loaded for
    the instance's lifetime. Injected into FilterDesign.q14_coefficients()
    rather than used as a module-level singleton (CONTRACTS.md §1).
    """

    def __init__(self, build_dir: str | Path | None = None) -> None:
        self._tmpdir_obj: tempfile.TemporaryDirectory | None = None
        if build_dir is None:
            self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="iir_filter_design_")
            build_dir = self._tmpdir_obj.name
        self.lib_path = compile_shared_library(Path(build_dir))
        self._lib = ctypes.CDLL(str(self.lib_path))
        _configure_argtypes(self._lib)

    def close(self) -> None:
        if self._tmpdir_obj is not None:
            self._tmpdir_obj.cleanup()
            self._tmpdir_obj = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def design_lp(self, fc: float, fs: float) -> Q14Coefficients:
        out = Q14Coeffs()
        rc = self._lib.filter_design_lp(fc, fs, ctypes.byref(out))
        _raise_for_rc(rc, "filter_design_lp")
        return Q14Coefficients(out.b0, out.b1, out.b2, out.a1, out.a2)

    def design_hp(self, fc: float, fs: float) -> Q14Coefficients:
        out = Q14Coeffs()
        rc = self._lib.filter_design_hp(fc, fs, ctypes.byref(out))
        _raise_for_rc(rc, "filter_design_hp")
        return Q14Coefficients(out.b0, out.b1, out.b2, out.a1, out.a2)

    def design_bp(self, f_low: float, f_high: float, fs: float) -> Q14Coefficients:
        out = Q14Coeffs()
        rc = self._lib.filter_design_bp(f_low, f_high, fs, ctypes.byref(out))
        _raise_for_rc(rc, "filter_design_bp")
        return Q14Coefficients(out.b0, out.b1, out.b2, out.a1, out.a2)

    def design_ap(self, fc: float, fs: float, q: float) -> Q14Coefficients:
        out = Q14Coeffs()
        rc = self._lib.filter_design_ap(fc, fs, q, ctypes.byref(out))
        _raise_for_rc(rc, "filter_design_ap")
        return Q14Coefficients(out.b0, out.b1, out.b2, out.a1, out.a2)

    def process_impulse(self, coeffs: Q14Coefficients, n_samples: int) -> list[int]:
        """Feeds a Q14 unit impulse (amplitude = Q14Coefficients.SCALE) through
        biquad_q14_process() sample-by-sample and returns the raw Q14 output
        samples. Drives the CONTRACTS.md §7 automated impulse-response test;
        not part of the app's own validation pipeline.
        """
        state = BiquadState()
        c_coeffs = Q14Coeffs(coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a1, coeffs.a2)
        self._lib.biquad_q14_init(ctypes.byref(state), ctypes.byref(c_coeffs))
        ys = []
        for n in range(n_samples):
            x = Q14Coefficients.SCALE if n == 0 else 0
            ys.append(int(self._lib.biquad_q14_process(ctypes.byref(state), ctypes.c_int32(x))))
        return ys
