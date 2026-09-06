# IIR Filter Design Suite

Interactive desktop suite for designing 2nd-order Butterworth IIR filter
chains (LP/HP/BP/AP) for Cortex-M4 firmware targets: PyQt6 UI, ideal
(float64) vs. Q14 fixed-point (compiled C) response comparison, and a
PDF + C-header + PNG export pipeline.

- Product vision: [`docs/CONCEPT.md`](docs/CONCEPT.md)
- Authoritative implementation contract (formulas, ABI, tolerances,
  resolved ambiguities): [`docs/CONTRACTS.md`](docs/CONTRACTS.md)

This document covers environment setup, running the app, and running the
test suite. Where anything here conflicts with `docs/CONTRACTS.md`, the
contract wins.

---

## 1. Requirements

- **Python 3.12 or newer** (developed/tested through 3.14).
- A **GCC- or Clang-compatible C compiler on PATH** at runtime — see
  [§3](#3-c-compiler-requirements) below. There is no degraded/mock mode:
  the app cannot run without a working native Q14 backend
  (`docs/CONTRACTS.md` §9).
- No PyInstaller/frozen build in v1 — this is a source / single-folder,
  `python main.py` delivery (`docs/CONTRACTS.md`, top of file).

## 2. Setting up a virtual environment

From the repository root:

```bash
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` pins the Python dependencies:

| Package | Purpose |
|---|---|
| `numpy` | Coefficient/array math |
| `scipy` | `freqz` evaluation and `dimpulse` (tests only) — never used as a filter *designer* (`docs/CONTRACTS.md` §4) |
| `matplotlib` | Embedded Bode plots (`ui/widgets/bode_widget.py`) and exported PNGs |
| `PyQt6` | Desktop UI |
| `reportlab` | PDF report generation (`export.py`) |

## 3. C compiler requirements

The native Q14 backend (`src/c/filter_design.c`, `src/c/biquad_q14.c`) is
compiled into a shared library **at every app launch** — recompiled fresh
each process start, not cached to disk across launches
(`docs/CONTRACTS.md` §9). Compiler discovery order: the `$CC` environment
variable (if set) → `gcc` → `cc` → `clang`.

**GCC or Clang only — MSVC is explicitly unsupported.**

### Linux

Install GCC or Clang from your distro's package manager, e.g.:

```bash
sudo apt install gcc      # Debian/Ubuntu
sudo dnf install gcc      # Fedora
```

Compiled with (`c_codegen.py`, matching `docs/CONTRACTS.md` §9 exactly):

```bash
gcc -O2 -fPIC -shared -o filter_design.so \
    src/c/filter_design.c src/c/biquad_q14.c -Isrc/c -lm
```

### Windows

Plain MSVC (`cl.exe`) is **not** supported — ctypes needs a standard cdecl
DLL without an MSVC-runtime dependency assumption. Two options work:

1. **MinGW-w64 GCC** (recommended) — e.g. via
   [MSYS2](https://www.msys2.org/) (`pacman -S mingw-w64-x86_64-gcc`) or a
   standalone MinGW-w64 toolchain. Ensure `gcc.exe` is on `PATH`.
2. **Clang in GNU-compatible mode** — `clang --target=x86_64-w64-mingw32`.
   Do **not** use `clang-cl` (MSVC-compatible mode); it assumes an
   MSVC-style runtime/ABI that ctypes' plain cdecl loading does not expect.

Compiled with (no `-fPIC` needed on Windows):

```bash
gcc -O2 -shared -o filter_design.dll ^
    src/c/filter_design.c src/c/biquad_q14.c -Isrc/c -lm
```

If no working compiler is found (or found but fails to compile), the app
shows a blocking **Retry**/**Quit** dialog with the compiler's stderr
before the main window ever appears (`docs/CONTRACTS.md` §9, §13) — there
is no way to proceed without a successful compile.

## 4. Running the app

```bash
python main.py
```

## 5. Running the tests

```bash
pytest
```

`pytest.ini` sets `pythonpath = src/python src`, so no extra `PYTHONPATH`
setup is needed. A working GCC/Clang on `PATH` is required for the native
(`test_native_*`, `test_export.py`, `test_firmware_harness.py`) tests —
`tests/conftest.py` compiles the shared library once per test session.

### Reproducible offscreen Qt test run

The UI tests (`test_ui_smoke.py`, `test_inspector.py`) construct real
`PyQt6.QtWidgets` objects and need a Qt platform plugin, but not a real
display. Force the headless `offscreen` plugin explicitly so the run is
reproducible in CI / over SSH / in any environment without a windowing
system:

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

(The test files themselves also set `QT_QPA_PLATFORM=offscreen` as a
default before importing PyQt6, so a plain `pytest` run is headless too —
the explicit environment variable above is the reproducible, explicit form
to use in CI configuration.)

## 6. Generated firmware header: coefficient-only by design

`export_design()` writes a per-design `export_YYYYMMDD_HHMMSS/` folder
containing `filter_design.h` — plain integer `#define FILT<n>_B0/.../A2`
literals plus `Q14_SCALE`/`Q14_TO_FLOAT` (`docs/CONTRACTS.md` §10). It is
intentionally **coefficient-only** and does not bundle `biquad_q14.h`/
`biquad_q14.c` into the export folder:

- CONCEPT.md §7's export directory listing enumerates exactly
  `report.pdf`, `filter_design.h`, `bode_combined.png`, `bode_<type>_<n>.png`,
  and `error_sweep_<n>.png` — no `biquad_q14.*` entry.
- `biquad_q14.{h,c}` is reference firmware source that lives once, in this
  repository's `src/c/` tree (`docs/CONTRACTS.md` §7) — duplicating it into
  every export folder would create N copies to keep in sync for no benefit;
  a firmware integrator combines the generated header with `src/c/biquad_q14.{h,c}`
  from this repository themselves, the same way `tests/test_firmware_harness.py`
  does for verification.

That combination is proven, not just asserted: `tests/test_firmware_harness.py`
compiles a small harness that `#include`s a freshly generated
`filter_design.h` together with `src/c/biquad_q14.h`, links against
`src/c/biquad_q14.c` with `gcc -Wall -Wextra -Werror`, **runs** the
resulting binary, and cross-checks its output against the same
`NativeBackend.process_impulse()` ctypes path used elsewhere in this suite.

## 7. Out of scope (v1)

Save/load, undo/redo, real-time audio preview, parallel (summing) topology,
filter orders other than 2nd, and filter families other than Butterworth
are all explicitly out of scope — see `docs/CONCEPT.md` §9 and
`docs/CONTRACTS.md` §13.
