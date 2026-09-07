# IIR Filter Design Suite — Concept Document

**Status:** Design phase  
**Forked from:** IIR-Compare  
**Date:** 2026-08-18  

> **Implementation-ready contracts:** exact formulas, interfaces, ABI, tolerances, and
> resolved ambiguities live in [`CONTRACTS.md`](./CONTRACTS.md), which is authoritative
> wherever it differs from the sketches in this document (notably the Band-Pass and
> All-Pass coefficient formulas below — see CONTRACTS.md §3).

---

## 1. Goal

An interactive desktop design suite for 2nd-order Butterworth IIR filters targeting
embedded Cortex-M4 devices. The user assembles a chain of filter blocks, inspects each
filter's Bode diagram (ideal vs Q14 C implementation), and exports a self-contained
deliverable (PDF report + C header + PNG plots) ready to drop into firmware.

---

## 2. Filter Types

All filters are **2nd-order Butterworth** unless noted. All support both a
**scipy-based ideal** computation and a **Q14 fixed-point C** computation.

| Block | Parameters | Notes |
|-------|-----------|-------|
| **Low-Pass (LP)** | `fc` [Hz] | Direct Form 1 biquad, bilinear transform |
| **High-Pass (HP)** | `fc` [Hz] | Same topology, numerator sign-flipped |
| **Band-Pass (BP)** | `f_low`, `f_high` [Hz] | `fc = √(f_low · f_high)`, `Q = fc / (f_high − f_low)` |
| **All-Pass (AP)** | `fc` [Hz], `Q` (damping) | Flat magnitude, adjustable phase. 2nd-order phase shifter; `Q` defaults to Butterworth damping. |
| **Peak (PK)** | `fc` [Hz], `Q`, `gain` [dB] | Parametric bell boost/cut. Not a Butterworth design — bilinear-transformed peaking-EQ prototype, pre-warped like LP/HP/BP. `|H(fc)|` = exactly `gain` dB; returns to 0 dB at DC and Nyquist. |

### Band-Pass parameterization
- User enters `f_low` and `f_high`; `fc` and `Q` are derived and displayed live.
- Bilinear-transform design of a true 2nd-order BP biquad (not LP+HP cascade).
  - Transfer function: `H(z) = b0(1 − z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)` → `b1 = 0`
  - This gives exactly -3 dB at `f_low` and `f_high`.

### All-Pass parameterization
- Flat magnitude (`|H(ejω)| = 1` for all ω), phase shifts from 0° (DC) to −360° (Nyquist).
- Phase = −180° exactly at `fc`.
- Numerator mirrors denominator: if denom = `1 + a1·z⁻¹ + a2·z⁻²`, then num = `a2 + a1·z⁻¹ + 1`.
- `Q` is user-adjustable because it controls the width and steepness of the phase transition while preserving unity magnitude.
- Default `Q = 1/√2 ≈ 0.707` (Butterworth damping). The UI constrains `Q` to
  `0.25 <= Q <= 4.0` to prevent invalid or impractical designs while retaining useful
  control over the phase transition.
- The selected `Q` is used consistently by the scipy ideal calculation, the C coefficient calculation, plots, validation, and export.

### Peak parameterization
- Parametric bell boost/cut around `fc`, with adjustable `Q` (bandwidth) and `gain` (dB).
- Not a Butterworth design (no such thing as a "Butterworth peak" filter) — derived from
  the analog peaking-EQ prototype `H(s) = (s² + (A/Q)s + 1) / (s² + s/(A·Q) + 1)`,
  `A = 10^(gain/40)`, bilinear-transformed with pre-warping `K = tan(π·fc/fs)` — the same
  prewarped family as LP/HP/BP (unlike All-Pass, which uses the un-prewarped digital `w0`).
- `|H(fc)|` = exactly the configured `gain` in dB; the response returns to exactly 0 dB at
  DC and at Nyquist, independent of `Q` or `gain`.
- `Q` range is `0.8 <= Q <= 4.0` (default `1.0`) — narrower than All-Pass's `[0.25, 4.0]`:
  at low `Q` combined with high `|gain|`, Peak's coefficients are not bounded by ~2.0 the
  way every other filter type's are (e.g. `Q=0.25` at `gain=+15 dB` would push a
  coefficient to ~3.11, well past what Q14 int16 storage can represent) — see
  CONTRACTS.md §5 for the full derivation. `gain` is constrained to
  `-15 dB <= gain <= 15 dB`, default `+6 dB`.
- The selected `Q`/`gain` are used consistently by the scipy ideal calculation, the C
  coefficient calculation, plots, validation, and export — same rule as All-Pass's `Q`.

---

## 3. Architecture

```
IIR-FilterDesignSuite/
├── main.py                        Entry point — launches PyQt6 app
├── requirements.txt               PyQt6, matplotlib, scipy, numpy, reportlab
├── setup_venv.py                  Cross-platform venv bootstrap (from IIR-Compare)
│
├── src/
│   ├── python/
│   │   ├── filters/
│   │   │   ├── base.py            Abstract FilterDesign (scipy ideal + Q14 coeff interface)
│   │   │   ├── lowpass.py         ButterworthLP
│   │   │   ├── highpass.py        ButterworthHP
│   │   │   ├── bandpass.py        ButterworthBP
│   │   │   ├── allpass.py         ButterworthAP
│   │   │   └── peak.py            PeakFilter (parametric EQ, not Butterworth)
│   │   ├── c_codegen.py           Compiles src/c/ → shared lib; exposes Q14 design fns
│   │   ├── error_analysis.py      Error sweep: ideal vs Q14 across full fc range
│   │   └── export.py              PDF report + C header (.h) + PNG export
│   │
│   ├── c/
│   │   ├── filter_design.h/.c     Q14 coefficient design: lp, hp, bp, ap
│   │   └── biquad_q14.h/.c        Generic Direct Form 1 Q14 biquad (state + process)
│   │
│   └── ui/
│       ├── app.py                 QMainWindow — three-panel layout
│       ├── palette.py             Left panel: draggable filter type buttons
│       ├── canvas.py              Center panel: design canvas (drop zone, chain view)
│       ├── inspector.py           Right panel: Bode plot + coefficient table
│       └── widgets/
│           ├── filter_block.py    Draggable/selectable filter block widget
│           └── bode_widget.py     Embedded matplotlib FigureCanvas (amplitude + phase)
│
└── tests/
    ├── conftest.py                Compiles shared lib once per session
    ├── test_coefficients.py       Q14 accuracy vs scipy for all 4 filter types
    ├── test_error_sweep.py        Max/RMS error across fc sweep per filter type
    └── test_bode.py               Bode response shape (passband, stopband, -3dB point)
```

---

## 4. UI Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  IIR Filter Design Suite          fs = [13333] Hz   [Export ▾]               │
├────────────┬───────────────────────────────────┬────────────────────────────┤
│  PALETTE   │         DESIGN CANVAS             │       INSPECTOR            │
│            │                                   │                            │
│  ┌──────┐  │  ┌──────────────────────────────┐ │  [Combined] [LP@3k] [HP]   │
│  │  LP  │  │  │                              │ │                            │
│  └──────┘  │  │  ┌─────┐   ┌─────┐          │ │  ┌──── Bode: LP@3kHz ────┐ │
│  ┌──────┐  │  │  │ LP  │──▶│ HP  │──▶ out   │ │  │  Amplitude (dB)       │ │
│  │  HP  │  │  │  │3kHz │   │8kHz │          │ │  │  ~~~Ideal             │ │
│  └──────┘  │  │  └─────┘   └─────┘          │ │  │  ---C Q14             │ │
│  ┌──────┐  │  │                              │ │  │                       │ │
│  │  BP  │  │  │  drag filter blocks here     │ │  │  Phase (°)            │ │
│  └──────┘  │  │  connect in series           │ │  └───────────────────────┘ │
│  ┌──────┐  │  └──────────────────────────────┘ │                            │
│  │  AP  │  │                                   │  Coefficients:             │
│  └──────┘  │  [Select All] [Clear] [Validate]  │  ┌────────┬───────┬──────┐ │
│            │                                   │  │        │ Ideal │ Q14  │ │
└────────────┴───────────────────────────────────┤  │ b0     │0.1023 │0.1021│ │
                                                 │  │ b1     │0.2046 │0.2044│ │
                                                 │  │ b2     │0.1023 │0.1021│ │
                                                 │  │ a1     │-1.358 │-1.357│ │
                                                 │  │ a2     │0.5916 │0.5916│ │
                                                 │  └────────┴───────┴──────┘ │
                                                 │                            │
                                                 │  response error:           │
                                                 │  max Δ = 0.012 dB          │
                                                 │  coefficient sweep: 1000  │
                                                 └────────────────────────────┘
```

### Panel behaviour

**Palette (left)**
- Five static draggable buttons: LP, HP, BP, AP, PK.
- Drag onto canvas to instantiate a block with default parameters.

**Design Canvas (center)**
- `fs` input at top — shared across all blocks.
- Filter blocks are connected left-to-right in series (chain topology).
- Each block shows its type and main parameter(s).
- Clicking a block selects it and updates the Inspector.
- A "Combined" view plots the cascade Bode (product of all transfer functions).
- Blocks can be reordered by drag, deleted with Delete key.

**Inspector (right)**
- Tab strip: "Combined" + one tab per filter block in the chain.
- Bode diagram: amplitude (dB) and phase (°, unwrapped) on two stacked subplots.
  - Two curves: **Ideal** (scipy) and **C Q14** (from compiled shared lib).
  - x-axis: log scale, 10 Hz to min(0.7·fs, fs/2−1).
  - y-axis: data-driven.
- Coefficient table: b0, b1, b2, a1, a2 — ideal (float) vs Q14 (int32_t / 2¹⁴).
- Selected-filter response validation: max and RMS amplitude error (dB) between the
  ideal and Q14 responses for the current design.
- C coefficient accuracy: a sweep of 1,000 cutoff frequencies over the complete
  supported cutoff range, comparing C-generated coefficients with ideal coefficients.
  Report max and RMS absolute coefficient error, plus the worst-case cutoff. This is
  reported separately from response error.

---

## 5. C Coefficient Design

All five filter types use the **bilinear transform** with pre-warping at fc.

```c
// shared precomputation (K = tan(π·fc/fs))
float K    = tanf(M_PI * fc / fs);
float K2   = K * K;
float norm = K2 + M_SQRT2 * K + 1.0f;   // Butterworth Q = 1/√2

// LP:   b0=b2=K²/norm,  b1=2·b0,   a1=2(K²-1)/norm, a2=(K²-√2K+1)/norm
// HP:   b0=b2=1/norm,   b1=-2·b0,  a1,a2 same as LP
// BP:   dual-edge-prewarped (K_low=tan(π·f_low/fs), K_high=tan(π·f_high/fs)) —
//       exact formula in CONTRACTS.md §3; a single center-K approximation does
//       NOT hit exact -3dB edges for non-narrow bands.
// AP:   selected Q is used to calculate the denominator; num = {a2, a1, 1},
//       den = {1, a1, a2} (mirror of the denominator) — exact w0/alpha formula
//       in CONTRACTS.md §3
// PK:   peaking EQ (not Butterworth) — bilinear transform of an analog bell
//       prototype with A = 10^(gain_dB/40), pre-warped like LP/HP/BP;
//       exact formula in CONTRACTS.md §3

// Q14 quantization:
int32_t q14(float x) { return (int32_t)roundf(x * 16384.0f); }
```

The C design functions (`filter_design.c`) are called at runtime (same as IIR-Compare).
The shared library is compiled once at app start, cached until the next launch.

---

## 6. Error Analysis

Validation has two separate analyses so response quality is not confused with the
accuracy of the C implementation's parameterization.

### Selected-filter response validation

For each filter block in the Inspector, the response comparison runs:

1. Compute the ideal Bode response for the current block parameters via scipy.
2. Compute Q14 coefficients via the C library and reconstruct its transfer function.
3. Compute amplitude error = `|ideal_dB − q14_dB|` over the plotted frequency grid.
4. Report max and RMS amplitude error in dB.

The "Validate" button in the canvas runs this response comparison for all blocks and
surfaces any block where max amplitude error > 0.1 dB with a warning badge. The same
comparison is also shown for the combined series cascade.

### C coefficient accuracy sweep

For each filter type, the suite independently evaluates 1,000 evenly spaced cutoff
frequencies from 10 Hz to `0.45·fs`, which is the complete supported cutoff range.
For LP, HP, and AP this is `fc`; for BP it is the center frequency `fc`, while the
configured bandwidth (or derived `Q`) remains fixed during the sweep. For each point
it:

1. Computes ideal floating-point coefficients using the same design equations and
  parameters (including the selected `Q` where applicable).
2. Computes C-generated Q14 coefficients and converts them back to floating point.
3. Calculates coefficient error for `b0`, `b1`, `b2`, `a1`, and `a2` as the absolute
  difference between C and ideal values.

The report includes maximum and RMS coefficient error across all coefficients and
cutoff points, the worst-case cutoff, and per-coefficient maxima. This sweep is
diagnostic; the 0.1 dB pass/fail threshold applies to amplitude response, not to
coefficient error.

---

## 7. Export / Delivery

Triggered by the **Export** button in the toolbar. Produces a timestamped output folder:

```
export_YYYYMMDD_HHMMSS/
├── report.pdf          Full report (see below)
├── filter_design.h     C header with Q14 coefficients for all blocks
├── bode_combined.png   Combined chain Bode plot
├── bode_<type>_<n>.png Individual Bode plot per block
└── error_sweep_<n>.png Error-vs-fc sweep plot per block
```

### PDF report contents

1. **Design summary** — fs, filter chain diagram, topology
2. **Per-filter section** — Bode plot, coefficient table (ideal + Q14), error sweep summary
3. **Combined chain** — combined Bode (ideal + Q14 cascade)
4. **C header listing** — inline copy of `filter_design.h`
5. **Test results** — pass/fail per block (max error < 0.1 dB threshold)

### C header format

```c
// Generated by IIR Filter Design Suite — 2026-08-18T10:30:00
// fs = 13333 Hz

// Filter 1: Butterworth Low-Pass  fc = 3000 Hz
#define FILT1_B0  Q14(0.10234f)   // 1677  (Q14 = 16384 * coeff)
#define FILT1_B1  Q14(0.20468f)   // 3354
#define FILT1_B2  Q14(0.10234f)   // 1677
#define FILT1_A1  Q14(-1.35822f)  // -22253
#define FILT1_A2  Q14(0.59158f)   // 9695
```

---

## 8. Key Design Decisions (inherited + extended)

| Decision | Choice | Reason |
|----------|--------|--------|
| Filter order | 2nd only | Single SOS section; simpler C, easier to validate |
| C arithmetic | Q14 fixed-point | Matches IIR-Compare baseline; runs on M0/M3/M4 without FPU |
| UI framework | PyQt6 + embedded matplotlib | All-Python; no JS build step; matplotlib handles Bode natively |
| Band-pass parameterization | f_low + f_high | Intuitive as passband edges; fc and Q shown as derived values |
| All-pass Q | User-adjustable, default `1/√2` | Q controls phase-transition steepness; unity magnitude is preserved; UI range `0.25 <= Q <= 4.0` |
| Chain topology | Series only | Parallel (summing) is out of scope for v1 |
| Export | PDF + C header + PNGs | PDF for documentation, header for firmware, PNGs for datasheets |
| Compilation | Runtime ctypes (subprocess gcc) | Same pattern as IIR-Compare; no build system needed |

---

## 9. Out of Scope (v1)

- Filter orders other than 2nd
- Filter families other than Butterworth (Chebyshev, Elliptic, Bessel)
- Parallel filter topology (summing junction)
- Bandpass as LP+HP cascade (use true 2nd-order BP biquad instead)
- Floating-point C implementation (Q14 only)
- Real-time audio preview
- Undo/redo history
- Save/load project file

---

## 10. Implementation Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| 1 | Core filter math | `src/python/filters/*.py` — ideal Bode for LP/HP/BP/AP |
| 2 | C Q14 coefficient design | `src/c/filter_design.c` + ctypes wrapper for all 4 types |
| 3 | Error analysis | `error_analysis.py` — fc sweep, max/RMS error per filter |
| 4 | UI skeleton | Three-panel PyQt6 window, static filter blocks |
| 5 | Drag-and-drop canvas | Drag from palette → instantiate block, connect chain |
| 6 | Inspector | Live Bode update + coefficient table on block select |
| 7 | Export | PDF + C header + PNG export pipeline |
| 8 | Tests | Pytest suite: coefficient accuracy, error sweep, Bode shape |