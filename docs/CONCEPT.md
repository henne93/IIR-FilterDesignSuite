# IIR Filter Design Suite — Concept Document

**Status:** Implemented (v1 shipped, and extended beyond the original v1 scope)  
**Forked from:** IIR-Compare  
**Date:** 2026-08-18 (original vision) — last revised 2026-09-08  

> **Implementation-ready contracts:** exact formulas, interfaces, ABI, tolerances, and
> resolved ambiguities live in [`CONTRACTS.md`](./CONTRACTS.md), which is authoritative
> wherever it differs from the sketches in this document (notably the Band-Pass and
> All-Pass coefficient formulas below — see CONTRACTS.md §3). CONTRACTS.md's
> "Summary of deviations" section is the single source of truth for everything that
> changed since this document was first written; this revision folds the
> user-facing consequences of those deviations back into the sections below.
> For setup and day-to-day usage, see [`README.md`](../README.md).

---

## 1. Goal

An interactive desktop design suite for 2nd-order IIR filters (Butterworth LP/HP/BP/AP,
plus a parametric Peak/EQ type) targeting embedded Cortex-M4 devices. The user assembles
a chain of filter blocks, inspects each filter's Bode diagram (ideal vs Q14 C
implementation), and exports a self-contained deliverable (PDF report, C header, PNG
plots, a project file, and a self-contained `source/` package) ready to drop into
firmware. Chains can also be saved to and reloaded from a project file for later editing.

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
├── requirements.txt               numpy, scipy, matplotlib, PyQt6, reportlab, pytest
│
├── src/
│   ├── python/
│   │   ├── filters/
│   │   │   ├── base.py            Abstract FilterDesign (scipy ideal + Q14 coeff interface)
│   │   │   ├── lowpass.py         ButterworthLP
│   │   │   ├── highpass.py        ButterworthHP
│   │   │   ├── bandpass.py        ButterworthBP
│   │   │   ├── allpass.py         ButterworthAP
│   │   │   ├── peak.py            PeakFilter (parametric EQ, not Butterworth)
│   │   │   └── chain.py           FilterChain — ordered series cascade, shared fs
│   │   ├── c_codegen.py           Compiles src/c/ → shared lib; exposes Q14 design fns (NativeBackend)
│   │   ├── error_analysis.py      Error sweep: ideal vs Q14 across full fc range
│   │   ├── export.py              PDF report + C header + PNGs + self-contained source/ package + C validation
│   │   └── project_file.py        Save/load a chain to/from a versioned .iirfilt JSON file
│   │
│   ├── c/
│   │   ├── filter_design.h/.c     Q14 coefficient design: lp, hp, bp, ap, pk
│   │   └── biquad_q14.h/.c        Generic Direct Form 1 Q14 biquad (state + process)
│   │
│   └── ui/
│       ├── app.py                 QMainWindow — three-panel layout, File/Edit menu bar (no toolbar)
│       ├── palette.py             Left panel: draggable filter type buttons
│       ├── canvas.py              Center panel: design canvas (drop zone, chain view)
│       ├── inspector.py           Right panel: Bode + gain plots, coefficient table, live metrics
│       └── widgets/
│           ├── filter_block.py       Draggable/selectable filter block widget
│           ├── bode_widget.py        Embedded matplotlib FigureCanvas (amplitude + phase)
│           ├── gain_widget.py        Embedded linear frequency/linear gain plot
│           └── measurement_cursor.py Hover cursor synced across one tab's plots
│
└── tests/                         pytest suite: per-filter-type math (test_lowpass.py, ...,
                                   test_peak.py), native Q14 ABI/coefficient/impulse/
                                   saturation tests, export + source-package round trips,
                                   project-file round trips, and offscreen UI smoke tests
                                   (see README.md §5 for how to run them)
```

---

## 4. UI Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  File   Edit                      IIR Filter Design Suite   fs = [13333] Hz  │
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
│  ┌──────┐  │  └──────────────────────────────┘ │  ┌── linear gain/freq ──┐ │
│  │  AP  │  │                                   │  └───────────────────────┘ │
│  └──────┘  │                                   │  Coefficients:             │
│  ┌──────┐  │                                   │  ┌────────┬───────┬──────┐ │
│  │  PK  │  │                                   │  │        │ Ideal │ Q14  │ │
│  └──────┘  │                                   │  │ b0     │0.1023 │0.1021│ │
└────────────┴───────────────────────────────────┤  │ b1     │0.2046 │0.2044│ │
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

`File` (Open, Save, Save As, Export) and `Edit` (Clear, Reset) are a menu bar,
not a toolbar — there is no dedicated toolbar row, and no "Select All" or
"Validate" button anywhere: validation is fully automatic (see below).

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
- **Clear** (Edit menu) empties the chain after a confirmation dialog; `fs` is
  left unchanged since it's a session-wide setting, not chain content.

**Inspector (right)**
- Tab strip: "Combined" + one tab per filter block in the chain.
- Bode diagram: amplitude (dB) and phase (°, unwrapped) on two stacked subplots.
  - Two curves: **Ideal** (scipy) and **C Q14** (from compiled shared lib).
  - x-axis: log scale, 10 Hz to min(0.7·fs, fs/2−1).
  - y-axis: phase fixed to ±180°; magnitude floor fixed at −100 dB (the top
    stays data-driven so filters with >0 dB gain, e.g. Peak boost, stay visible).
- A second, linear frequency/linear-gain plot sits below the Bode plot; a
  shared hover cursor synchronizes a frequency marker across both plots in
  the active tab, showing magnitude/phase/gain at the pointer.
- Coefficient table: b0, b1, b2, a1, a2 — ideal (float) vs Q14 (int16_t / 2¹⁴).
- Selected-filter response validation: max and RMS amplitude error (dB) between the
  ideal and Q14 responses for the current design.
- All of the above recomputes automatically on every parameter/chain edit —
  there is no "Validate" action anywhere in the UI.
- C coefficient accuracy: a sweep of 1,000 cutoff frequencies over the complete
  supported cutoff range, comparing C-generated coefficients with ideal coefficients.
  Report max and RMS absolute coefficient error, plus the worst-case cutoff. This is
  reported separately from response error.

---

## 5. C Coefficient Design

All five filter types use the **bilinear transform**. LP, HP, BP, and PK
pre-warp their critical frequency/frequencies (`K = tan(π·f/fs)`); AP is the
one exception — its defining properties (unity magnitude, `-180°` at `fc`)
hold for the un-prewarped digital `w0 = 2π·fc/fs` directly, so it skips
pre-warping (see CONTRACTS.md §3).

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

// Q14 quantization (storage is int16_t — narrowed from an earlier int32_t
// design; see CONTRACTS.md §7 for the domain-sweep headroom analysis):
int16_t q14(float x) { return (int16_t)roundf(x * 16384.0f); }
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

This response comparison re-runs automatically for every block on each parameter/chain
edit (there is no manual "Validate" action) and surfaces any block where max amplitude
error > 0.1 dB with a warning badge. The same comparison is also shown for the combined
series cascade.

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

Triggered by **File ▸ Export** (a plain menu action, not a toolbar button).
Produces a timestamped output folder:

```
export_YYYYMMDD_HHMMSS/
├── design.iirfilt              Current chain config (project_file.save_project())
├── source/                     Self-contained C deliverable (see below)
│   ├── biquad_q14/
│   │   ├── cfg/                 Reserved for future user-config macros (empty)
│   │   ├── inc/                 biquad_q14.h, filter_design_calc.h
│   │   ├── src/                 biquad_q14.c, filter_design_calc.c
│   │   └── gen/filter_design.h  GENERATED per-design Q14 coefficients
│   ├── app_template/example.c  Generated cascade-wiring demo
│   └── README.md                Integration instructions
└── reports/
    ├── biquad_q14_report.pdf   Full report (see below)
    ├── figures/
    │   ├── bode_combined.png    Combined chain Bode plot
    │   ├── bode_<type>_<n>.png  Individual Bode plot per block
    │   └── error_sweep_<n>.png Error-vs-fc sweep plot per block
    └── test/test_summary.txt   C compile+run validation result (see below)
```

`design.iirfilt` is a fixed filename every time, independent of whatever
project file the user separately has open/saved via File ▸ Save — so this
export directory round-trips to exactly the chain state that was exported.

### PDF report contents

1. **Design summary** — fs, filter chain diagram, topology
2. **Per-filter section** — Bode plot, coefficient table (ideal + Q14), error sweep summary
3. **Combined chain** — combined Bode (ideal + Q14 cascade)
4. **C header listing** — inline copy of the generated `filter_design.h`
5. **Test results** — pass/fail per block (max error < 0.1 dB threshold)

### C header format

Plain integer literals, not a `Q14(...)` macro — so the header compiles
standalone without the firmware consumer having to define anything first
(see CONTRACTS.md §10 for why this revises the original sketch):

```c
// Generated by IIR Filter Design Suite — 2026-08-18T10:30:00+02:00
// fs = 13333 Hz

#define Q14_SCALE 16384
#define Q14_TO_FLOAT(x) ((float)(x) / Q14_SCALE)

// Filter 1: Butterworth Low-Pass  fc = 3000 Hz
#define FILT1_B0  1677    // 0.10233688f (Q14, scale=16384)
#define FILT1_B1  3354    // 0.20467377f
#define FILT1_B2  1677    // 0.10233688f
#define FILT1_A1  -22253  // -1.35822105f
#define FILT1_A2  9695    // 0.59158379f
```

### The `source/` package

Every export also writes a `source/` subfolder — a complete, self-contained
C deliverable for an external firmware project, needing no other file from
this repository: `biquad_q14/` groups every filter source file into `cfg/`
(reserved, empty), `inc/` (headers), `src/` (implementation), and `gen/`
(the generated, per-design coefficient header); a sibling `app_template/`
holds a generated `example.c` wiring the exported chain's blocks in series,
and a `README.md` carries integration instructions, including the canonical
compile command (`-I biquad_q14/inc` is required since `app_template/` is a
sibling of `biquad_q14/`, not its parent). See README.md §6 for details.

### C validation (`reports/test/test_summary.txt`)

Every export also compiles and runs the just-written `source/` package
(with the same `gcc` used for the native backend) and cross-checks its
output against this export's own snapshot data — a cascade compile+run
check and a design-function check, both recorded as a PASS/FAIL/SKIPPED
verdict in `reports/test/test_summary.txt`. This step never blocks or fails
export itself (a missing compiler, or a compile/run/mismatch failure, is
recorded in the summary, not raised) — see CONTRACTS.md §10.

---

## 8. Key Design Decisions (inherited + extended)

| Decision | Choice | Reason |
|----------|--------|--------|
| Filter order | 2nd only | Single SOS section; simpler C, easier to validate |
| C arithmetic | Q14 fixed-point, `int16_t` storage | Runs on M0/M3/M4 without FPU; narrowed from an initial `int32_t` design to keep the firmware-facing biquad 16-bit end to end (matches a 16-bit ADC/DAC pipeline) — see CONTRACTS.md §7 |
| UI framework | PyQt6 + embedded matplotlib | All-Python; no JS build step; matplotlib handles Bode natively |
| UI chrome | File/Edit menu bar, no toolbar | File: Open/Save/Save As/Export; Edit: Clear/Reset — replaced an earlier toolbar-based sketch |
| Band-pass parameterization | f_low + f_high | Intuitive as passband edges; fc and Q shown as derived values |
| All-pass Q | User-adjustable, default `1/√2` | Q controls phase-transition steepness; unity magnitude is preserved; UI range `0.25 <= Q <= 4.0` |
| Peak filter | Added as a 5th, non-Butterworth type | Parametric bell boost/cut for EQ use cases; narrower `Q` range (`0.8–4.0`) to keep coefficients within int16 headroom at extreme gain |
| Chain topology | Series only | Parallel (summing) is out of scope for v1 |
| Validation | Fully automatic, no "Validate" action | Every parameter/chain edit recomputes response error and coefficient-sweep metrics immediately |
| Persistence | Save/Open a versioned `.iirfilt` JSON project file | Reverses the original "no save/load in v1" decision; round-trips the full chain (including invalid/disabled blocks) |
| Export | PDF + C header + PNGs + project file + self-contained `source/` package + C validation | PDF for documentation, header for firmware, PNGs for datasheets, `source/` for a zero-dependency drop-in, `design.iirfilt` so the export round-trips, a compile+run check for real confidence |
| Compilation | Runtime ctypes (subprocess gcc) | Same pattern as IIR-Compare; no build system needed |

---

## 9. Out of Scope (v1)

- Filter orders other than 2nd
- Filter families other than Butterworth (Chebyshev, Elliptic, Bessel) —
  Peak/EQ was added later (§2) as a 5th block type within the same
  2nd-order/Q14 framework, not as a new filter *family* in this sense
- Parallel filter topology (summing junction)
- Bandpass as LP+HP cascade (use true 2nd-order BP biquad instead)
- Floating-point C implementation (Q14 only)
- Real-time audio preview
- Undo/redo history

**No longer out of scope:** save/load project files shipped after v1's initial
cut (§8, §10) — see CONTRACTS.md §13/§15 for the format and UI rules.

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

**Delivered beyond the original v1 plan:**

| Milestone | Deliverable |
|-----------|-------------|
| Int16 Q14 storage | Coefficients/state narrowed from `int32_t` to `int16_t`, saturating 32-bit accumulator (CONTRACTS.md §7) |
| Source package export | Self-contained `source/` subfolder per export, proven to compile/run standalone (`tests/test_source_package.py`), plus a production C validation step recorded in `reports/test/test_summary.txt` |
| Project save/load | Versioned `.iirfilt` JSON round-trip (`project_file.py`, `tests/test_project_file.py`) |
| Fixed Bode y-axes | Phase locked to ±180°, magnitude floor locked to −100 dB, in both the GUI and exported plots |
| File/Edit menu bar | Replaced the original toolbar sketch; Open/Save/Save As/Export under File, Clear/Reset under Edit |
| Peak (PK) filter type | Parametric bell EQ, a 5th block type (§2), including its own coefficient-domain int16-headroom analysis |

---

## 11. Time-Domain View (Feature Extension)

**Status:** Concept — in development on branch `feature/time-domain-view`.

### 11.1 Goal

Alongside the frequency-domain Bode view, let the user see the filter chain's
effect on an actual time-domain signal: a user-built source signal, filtered
through the chain both ways (ideal float and Q14 fixed-point), plotted
together. This answers "what does this filter actually do to a signal" in a
way a Bode plot cannot.

### 11.2 Signal Generator

The source signal is assembled from **additive building blocks**, analogous
in spirit to the filter chain but summed rather than cascaded (order does not
matter for a sum):

| Block | Parameters | Notes |
|-------|-----------|-------|
| **Sine** | frequency [Hz], amplitude, phase | Amplitude is normalized (§11.4) |
| **DC** | value | Constant offset, tests DC handling |
| **Noise** | amplitude (AWGN, additive white Gaussian) | Just another block — no special "clean vs. noisy" distinction; see below |
| **CSV Import** | file (time [s], value columns) | Resampled to the global `fs` (§11.3); raw file values are not assumed pre-normalized, hence the `factor` below |

Every block, regardless of type, additionally has a **`factor`** parameter
(default `1.0`) that multiplies its raw output before summation — a uniform
gain knob independent of the block's own amplitude/value/DC field. This
matters most for **CSV Import**, which has no native "amplitude" parameter
of its own (its values come straight from the file, in whatever units it was
recorded in) — `factor` is the only way to scale it into range. For the
other block types it's a convenience on top of their own amplitude field
(e.g. quickly halve a Sine's contribution without recomputing its amplitude).
Effective contribution of a block = `factor × (its own signal)`.

Any number of blocks (including multiple of the same type, e.g. two Sine
blocks at different frequencies) can be added; the **source signal is their
sum**. There is deliberately **no separate "original" vs. "noisy" signal** —
noise is just one more addable block, and the plot shows exactly one source
signal (whatever blocks the user assembled, noise included or not) alongside
the filtered results. This is a deliberate simplification agreed with the
user: distinguishing "clean" and "noisy" would require tagging blocks as
noise-vs-not and computing two source curves, which adds UI complexity the
user explicitly decided is unnecessary — the filtered-vs-source comparison
already shows the filter's effect.

### 11.3 Sample Rate & Duration

- Reuses the chain's existing global `fs` (same value shown at the top of the
  Design view) — there is exactly one `fs` in the app, so the time-domain
  signal is generated and filtered at that rate, guaranteeing consistency
  with the filter design.
- A **duration** field (ms, default 50 ms) controls the plotted window;
  sample count = `round(duration_ms / 1000 * fs)`.
- CSV import: two columns, time [s] and value. Values are linearly
  interpolated onto the uniform `fs` grid over the plotted duration (i.e.
  resampled to fit the global `fs`, not assumed to already match it).

### 11.4 Amplitude Convention & Q14 Scaling

Sine/DC/Noise amplitudes and CSV values are **normalized floats**, `-1.0` to
`+1.0` representing full scale, further scaled per-block by that block's
`factor` (§11.2). This keeps signal-block configuration unitless and
intuitive.

Note this is a *sample* scaling, distinct from the existing Q14
*coefficient* scaling (`× 16384`, §5) — `biquad_q14_process()` takes/returns
plain `int16_t` samples (§11.5), so the source signal must be mapped from
the normalized float domain to int16 counts before the Q14 path, and mapped
back for plotting.

That mapping is **user-configurable**, not a fixed `× 32767`: a **"Full-scale
reference"** control (int16 counts per normalized `1.0`, default `32767`)
sits alongside the duration/`fs` controls in the Inspector. Lowering it below
`32767` leaves headroom and lets the user deliberately explore/demonstrate
Q14 quantization and saturation behavior (e.g. a signal that's supposed to
represent a real ADC/DAC full-scale narrower than the int16 range); raising
it (up to `32767`, the hard int16 ceiling) uses the full available dynamic
range. The same reference value scales the Q14 output back down to the
normalized float axis for plotting, so source, ideal-filtered, and
Q14-filtered curves stay on one shared, comparable axis regardless of the
chosen full-scale reference. Exact clipping/rounding rules at this mapping
step are an implementation contract, to be pinned down in CONTRACTS.md
alongside the existing Q14 coefficient rules.

### 11.5 Filtering

Two filtered curves are computed from the same source signal, mirroring the
Ideal-vs-Q14 pairing already used for Bode/coefficients:

- **Ideal**: cascade each block's `ideal_coefficients()` (float `b`/`a`) in
  series over the source signal (scipy `lfilter`/`sosfilt`).
- **Q14**: cascade each block's Q14 coefficients using the existing
  `NativeBackend.process_samples()` C path (§3, `c_codegen.py`) — the output
  int16 stream of one block feeds the next block's input, matching how the
  cascade runs on the target firmware.

### 11.6 Scope: Combined + Per-Block

Like the Bode Inspector, the time-domain Inspector has a tab strip:
**"Combined"** (whole chain) plus **one tab per filter block**, so the user
can inspect the time-domain effect of an individual block as well as the
full cascade — consistent with the rest of the suite rather than a
combined-only special case.

### 11.7 UI: A Second Top-Level View

The app gains a top-level **view switch** (Design / Time Domain) that swaps
the entire central widget, rather than adding a 4th panel squeezed into the
existing 3-panel Design layout. The Time-Domain view **mirrors the existing
3-panel pattern**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  File   Edit          [ Design | Time Domain ]     IIR Filter Design Suite   │
├────────────┬───────────────────────────────────┬────────────────────────────┤
│  PALETTE   │      SIGNAL CANVAS                │       INSPECTOR            │
│ (signal    │                                    │                            │
│  blocks)   │  ┌─────┐  ┌─────┐  ┌─────┐        │  [Combined] [LP@3k] [HP]   │
│ ┌────────┐ │  │ Sine│  │ DC  │  │Noise│  Σ →   │  ┌── Time domain ────────┐ │
│ │  Sine  │ │  │1kHz │  │ 0.0 │  │0.05 │        │  │  Source               │ │
│ └────────┘ │  └─────┘  └─────┘  └─────┘        │  │  ~~~Ideal filtered    │ │
│ ┌────────┐ │                                    │  │  ---Q14 filtered      │ │
│ │  DC    │ │  drop signal blocks here            │  └───────────────────────┘ │
│ └────────┘ │  (summed, order doesn't matter)     │  duration: [ 50] ms        │
│ ┌────────┐ │                                    │  fs: 13333 Hz (from Design)│
│ │ Noise  │ │                                    │  Q14 full-scale: [32767]   │
│ └────────┘ │                                    │                            │
│ ┌────────┐ │                                    │                            │
│ │  CSV   │ │                                    │                            │
│ └────────┘ │                                    │                            │
└────────────┴───────────────────────────────────┴────────────────────────────┘
```

- **Palette (left)**: draggable signal-block buttons — Sine, DC, Noise, CSV
  Import — same drag-to-instantiate interaction as the filter palette. Each
  instantiated block also gets the common `factor` field (§11.2) alongside
  its own parameters.
- **Signal Canvas (center)**: dropped blocks shown as a flat, unordered
  collection feeding a sum (Σ) — no series connectors, since summation order
  doesn't matter (unlike the filter chain's series canvas). Blocks are
  selectable (to edit parameters) and deletable, same as filter blocks.
- **Inspector (right)**: tab strip (Combined + per-block), time-domain plot
  (source / ideal-filtered / Q14-filtered), duration field, read-only `fs`
  echoed from the Design view, and the Q14 full-scale reference (§11.4).
- The chain of **filter** blocks itself is unchanged and still edited only in
  the Design view; the Time-Domain view consumes it read-only via the same
  `FilterChain` model.

### 11.8 Out of Scope (this feature)

- Real-time / streaming audio playback of source or filtered signal
- Per-sample interactive scrubbing beyond the existing hover-cursor pattern
- Exporting time-domain plots/data as part of File ▸ Export (may follow
  later, not part of this iteration)
- Noise/signal-block parameters participating in the `.iirfilt` project file
  (persistence of the signal chain) — TBD, likely a later iteration once the
  core view works