# IIR Filter Design Suite

Interactive desktop suite for designing 2nd-order IIR filter chains
(Butterworth Low-Pass, High-Pass, Band-Pass, All-Pass, plus a parametric
Peak/EQ filter) for Cortex-M4 firmware targets. PyQt6 UI, ideal (float64) vs.
Q14 fixed-point (compiled C) response comparison, and a PDF + C-header + PNG
export pipeline with a self-contained `firmware/` drop-in package.

- Product vision: [`docs/CONCEPT.md`](docs/CONCEPT.md)
- Authoritative implementation contract (exact formulas, ABI, tolerances):
  [`docs/CONTRACTS.md`](docs/CONTRACTS.md) — wins wherever it and this
  README disagree.
- Exploratory architecture note, not implemented:
  [`docs/adaptive_bandpass_design.md`](docs/adaptive_bandpass_design.md)

## Requirements

- Python 3.12+ (developed/tested through 3.14).
- A GCC- or Clang-compatible C compiler on `PATH` — the native Q14 backend is
  compiled at every app launch, and there is no mock/degraded mode without it.
  **MSVC is not supported** (on Windows, use MinGW-w64 GCC or Clang in
  GNU-compatible mode — not `clang-cl`).
- No frozen/PyInstaller build — run from source with `python main.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` covers `numpy`, `scipy`, `matplotlib`, `PyQt6`, `reportlab`
(runtime), and `pytest` (for the test suite below).

Linux: `sudo apt install gcc` (or `dnf install gcc`). Windows: install
MinGW-w64 GCC (e.g. via [MSYS2](https://www.msys2.org/)) and ensure
`gcc.exe` is on `PATH`. If no working compiler is found, the app shows a
blocking Retry/Quit dialog with the compiler's error output on startup.

## Running

```bash
python main.py
```

The **File** menu holds Open / Save / Save As / Export; **Edit** holds Clear
/ Reset (no toolbar). A filter chain can be saved to and reloaded from a
versioned `.iirfilt` project file.

## Testing

```bash
pytest
```

`pytest.ini` already sets `pythonpath`, so no extra setup is needed. Native
tests need a working compiler on `PATH` (the shared library is compiled once
per test session). For a reproducible headless run (CI, SSH, no display):

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

## Export

**File ▸ Export** writes a timestamped `export_YYYYMMDD_HHMMSS/` folder
containing a PDF report, a coefficient-only `filter_design.h`, Bode/error PNG
plots, and a `firmware/` subfolder — a complete drop-in C package (coefficient
header, biquad implementation, the Q14 design-function implementation itself
for runtime recomputation, a generated usage example, and its own README)
that compiles standalone with no other file from this repository. See
`docs/CONTRACTS.md` §10 for the exact formats.

## Out of scope (v1)

Undo/redo, real-time audio preview, parallel (summing) topology, filter
orders other than 2nd, and filter families other than Butterworth (Peak/EQ
is the one non-Butterworth exception — see `docs/CONCEPT.md` §2/§9).
