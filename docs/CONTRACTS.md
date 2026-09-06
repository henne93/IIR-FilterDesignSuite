# IIR Filter Design Suite — Implementation Contracts

**Status:** Specification phase — no application code exists yet.
**Relationship to CONCEPT.md:** CONCEPT.md is the product vision and UI sketch. This
document is the authoritative, implementation-ready contract. Where the two disagree,
**this document wins**; the discrepancies are called out explicitly below with rationale.
**Scope:** Python 3.12+, Linux and Windows, source / single-folder delivery (no
PyInstaller/frozen build in v1), a GCC- or Clang-compatible C compiler required at
runtime on the target machine.

---

## 0. How to read this document

Thirteen contract areas, each with: the resolved decision, the reasoning, and (where
real ambiguity remains) an explicit open question. Section 14 gives module ownership,
build order, and required tests per phase — pulled forward into the chat report as well.

All formulas below were numerically verified (bilinear-transform algebra + direct
evaluation of H(z) at the claimed −3 dB / phase points) before being written down here.
See "Verification" note at the end of §3.

**Precision note (applies throughout this document):** the exact Butterworth edge value
is **−3.0103 dB** (`20·log10(1/√2)`). "−3 dB" is used elsewhere in this document, in the
UI, and in CONCEPT.md as commonly-understood shorthand for that exact value — it is never
literally `−3.000 dB`. Wherever a test, tolerance, or numeric assertion needs the precise
reference, use `−3.0103 dB`, not `−3 dB`.

---

## 1. Python filter-model interfaces and return types

```python
# src/python/filters/base.py

@dataclass(frozen=True)
class Coefficients:
    """H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2). a0 is always 1."""
    b0: float; b1: float; b2: float; a1: float; a2: float

    def as_ba(self) -> tuple[list[float], list[float]]:
        """Returns (b, a) exactly as scipy.signal.freqz expects."""
        return [self.b0, self.b1, self.b2], [1.0, self.a1, self.a2]

@dataclass(frozen=True)
class Q14Coefficients:
    """Q14 fixed-point, scale = 16384. Storage width is int32_t (see §7)."""
    b0: int; b1: int; b2: int; a1: int; a2: int
    SCALE: ClassVar[int] = 16384

    def to_float(self) -> Coefficients:
        s = self.SCALE
        return Coefficients(self.b0/s, self.b1/s, self.b2/s, self.a1/s, self.a2/s)

@dataclass(frozen=True)
class FrequencyResponse:
    freq_hz: np.ndarray        # (N,) strictly increasing, float64
    magnitude_db: np.ndarray   # (N,) float64
    phase_deg: np.ndarray      # (N,) float64, unwrapped

class FilterDesign(ABC):
    kind: ClassVar[Literal["LP", "HP", "BP", "AP"]]

    def __init__(self, fs: float, **params: float) -> None: ...

    @abstractmethod
    def ideal_coefficients(self) -> Coefficients: ...
    @abstractmethod
    def q14_coefficients(self, backend: "NativeBackend") -> Q14Coefficients: ...

    def ideal_response(self, freq_hz: np.ndarray) -> FrequencyResponse: ...
    def q14_response(self, freq_hz: np.ndarray, backend: "NativeBackend") -> FrequencyResponse: ...

    def validate(self) -> None:
        """Raises ValueError with a human-readable message. Never clamps silently."""

    def derived_params(self) -> dict[str, float]:
        """e.g. BP -> {'fc': ..., 'Q': ...}. Empty dict for LP/HP. AP -> {} (Q is a primary param)."""
```

Concrete subclasses: `LowPassFilter`, `HighPassFilter`, `BandPassFilter`, `AllPassFilter`
in `lowpass.py` / `highpass.py` / `bandpass.py` / `allpass.py`, each implementing
`ideal_coefficients()` per the closed-form equations in §3, and `q14_coefficients()` by
calling the matching native function (§8) through a shared `NativeBackend` handle
(injected, not a module-level singleton — keeps the filter classes testable without a
compiled library present).

**Rule:** all dataclass fields are plain Python `float`/`int`, never `numpy.float64` /
`numpy.int32` — numpy scalars are converted at the boundary (`float(x)`, `int(x)`) so
these types stay directly JSON- and repr-serializable for export and tests.

---

## 2. Coefficient ordering and denominator sign convention

- **Order, everywhere** (tuples, arrays, table columns, header defines, function
  signatures): `b0, b1, b2, a1, a2`.
- **a0 is always normalized to 1** and is never stored or displayed.
- **Denominator convention:** `1 + a1·z⁻¹ + a2·z⁻²` — positive signs on `a1`, `a2`. This
  is the same convention scipy uses internally (`sos` rows are `[b0,b1,b2,a0,a1,a2]`
  with positive `a1,a2`; `scipy.signal.freqz(b, a)` expects the same). Concretely:
  `scipy.signal.freqz([b0,b1,b2], [1,a1,a2])` — **no sign flip anywhere in this
  codebase.** This convention matters because some textbooks define the denominator as
  `1 − a1·z⁻¹ − a2·z⁻²` (opposite sign); mixing conventions is the single most common
  source of biquad bugs and must not happen here.

---

## 3. Exact LP, HP, BP, AP reference behavior

These are the **single source of truth**. The Python "ideal" path and the C "Q14" path
both implement exactly these equations — see §4 for why that equivalence is load-bearing.

The properties below (exact edge dB, exact phase points) are **mathematical identities**
of the formulas, holding for the full continuous Nyquist domain (`0 < fc < fs/2`, and
`0 < f_low < f_high < fs/2` for BP). §5 separately restricts the range of values a user
may actually enter to `[100 Hz, fc_max(fs)]` — a narrower, product-level constraint, not
a limitation of the math itself.

### Low-Pass
```
K    = tan(π·fc/fs)
norm = 1 + √2·K + K²
b0 = b2 = K²/norm
b1     = 2·b0
a1 = 2·(K²−1)/norm
a2 = (K² − √2·K + 1)/norm
```
Property: `|H(e^{j2πfc/fs})| = −3.0103 dB` exactly, for any `0 < fc < fs/2`.

### High-Pass
```
K, norm as above (same fc, fs)
b0 = b2 = 1/norm
b1 = −2·b0
a1, a2 = same as LP
```
Property: `−3.0103 dB` at `fc` exactly (same derivation, matched-Z-transformed
Butterworth prototype with the high-pass numerator).

### Band-Pass — **revised from CONCEPT.md**
CONCEPT.md's pseudocode (`b0=-b2=BW·K/norm_bp`, `a2=(K²-Kw·K+1)/norm_bp`) has an
undefined term (`Kw`) and, more importantly, only prewarps the center frequency —
verified numerically to give **-4.16 dB / -5.77 dB** (not −3/−3) at the band edges for a
2000–4000 Hz band on a 13333 Hz `fs`. The exact fix is to prewarp **both edges**
independently (half-angle tangent, same prewarping family as LP/HP) rather than
prewarping only the center and treating bandwidth linearly:
```
K_low  = tan(π·f_low/fs)
K_high = tan(π·f_high/fs)
wc2    = K_low · K_high          # squared prewarped center
BW     = K_high − K_low          # prewarped bandwidth
norm   = 1 + BW + wc2
b0 = BW/norm
b1 = 0
b2 = −BW/norm
a1 = 2·(wc2 − 1)/norm
a2 = (1 − BW + wc2)/norm
```
Displayed/derived values (unchanged from CONCEPT.md): `fc = √(f_low·f_high)`,
`Q = fc/(f_high−f_low)` — these are *reporting* quantities only; they are not inputs to
the coefficient formula above (which uses `f_low`/`f_high` directly).
Property: verified numerically exact `−3.0103 dB` at both `f_low` and `f_high`,
independent of bandwidth (checked from a 1-octave band up to a band spanning nearly the
entire valid range).

### All-Pass — **formula made explicit** (CONCEPT.md only described the structure)
```
w0    = 2π·fc/fs
alpha = sin(w0)/(2·Q)
norm  = 1 + alpha
a1 = −2·cos(w0)/norm
a2 = (1 − alpha)/norm
b0 = a2,  b1 = a1,  b2 = 1        # numerator mirrors denominator, per CONCEPT.md
```
Property: `|H(e^{jω})| = 1` for **all** ω — this is a structural identity of the
mirrored-numerator biquad, true for any stable `(a1, a2)`, independent of `Q`. Phase
crosses `−180°` exactly at `fc`, verified numerically to floating-point precision.
No prewarping is needed here (unlike BP) because the defining properties (unity
magnitude, `-180°` at `fc`) hold for the un-prewarped digital `w0` directly.

**Verification method:** all four formulas above were checked with a standalone
complex-arithmetic evaluation of `H(e^{jω})` (not scipy) at the claimed critical
frequencies, confirming `−3.0103 dB` (i.e. exact `−3 dB`) at every claimed edge and
exact `±0°`/`−180°` phase properties. Anyone re-implementing this in C should hold the
Python reference to the same numeric check as a unit test (see §14).

---

## 4. scipy vs. C comparison rules

**Critical resolution:** "ideal" does **not** mean `scipy.signal.butter()` /
`scipy.signal.bilinear()`. If the Python "ideal" path used scipy's own Butterworth
design routines while the C path used the closed-form equations in §3, the two would
diverge *before* any quantization is applied — which would corrupt the coefficient
accuracy sweep (§11) into measuring "which design algorithm did we pick" instead of
"how much error did Q14 quantization introduce."

- **"Ideal" coefficients** = the §3 closed-form equations evaluated in Python
  `float64` (numpy), full stop. scipy is used only as the frequency-response evaluator
  (`scipy.signal.freqz`), never as a filter designer.
- **"C Q14" coefficients** = the same §3 equations evaluated in C `float` (32-bit,
  matching CONCEPT.md's `tanf`/`float` usage), then Q14-quantized (§7).
- **Response comparison procedure**, both curves plotted from the same frequency grid
  (§6):
  1. Ideal: `freqz(b_ideal_f64, a_ideal_f64, worN=2π·f_grid/fs)`.
  2. Q14: dequantize (`Q14Coefficients.to_float()`), then
     `freqz(b_q14_f64, a_q14_f64, worN=2π·f_grid/fs)` — **same freqz call, same grid**,
     so the only source of divergence between the two curves is float32-rounding +
     Q14 quantization, not an evaluation-method mismatch.
  3. Amplitude error at each grid point = `|20·log10|H_ideal| − 20·log10|H_q14||` (dB).
- **Coefficient error** (§11 sweep) = `|ideal_coef − dequantized_q14_coef|`, absolute,
  per coefficient — not relative, not dB.
- **Deliberate non-simplification:** the "ideal" reference is float64 while the C path
  is float32-then-quantized. This is intentional, not a bug to "fix" later — real
  Cortex-M4 firmware also computes in float32 (or fixed-point), so the measured error
  is representative of real deployed behavior, not an artifact of comparing incompatible
  precisions.

---

## 5. Supported parameter ranges and invalid-input behavior

| Parameter | Range | Notes |
|---|---|---|
| `fs` | `5,000 Hz ≤ fs ≤ 40,000 Hz` | Final product decision — replaces the earlier `1 kHz`–`1 MHz` placeholder that had no basis in CONCEPT.md. |
| `fc` (LP, HP, AP) | `100 Hz ≤ fc ≤ fc_max(fs)` | See "Effective digital cutoff maximum" below. **Deviation from CONCEPT.md §6**, which describes the sweep range as "10 Hz to `0.45·fs`" — see Summary of deviations, item 6. |
| `f_low`, `f_high` (BP) | each independently in `[100, fc_max(fs)]`, and `f_low < f_high` strictly | Both edges are bounded by the same effective range as `fc` above. `fc`/`Q` are derived and always land in-range automatically given that constraint. |
| `Q` (AP) | `0.25 ≤ Q ≤ 4.0` | Per CONCEPT.md. |

**Effective digital cutoff maximum:**
```
fc_max(fs) = min(20,000 Hz, 0.45·fs)
```
The `20,000 Hz` term is the absolute ceiling on any user-facing cutoff/edge value
(upper edge of nominal audio bandwidth). The `0.45·fs` term keeps the design comfortably
inside the Nyquist limit (`fs/2`) so the bilinear prewarping (`tan(π·fc/fs)`) never
approaches its pole at `fc = fs/2`. This single `fc_max(fs)` expression is the one
formula used everywhere a "design cutoff upper bound" is needed (§6 sweep grid, §11
tolerance-edge language, §14 test domain) — there is no second definition.

**Note (not currently load-bearing, flagged for the user):** for every `fs` in the
supported `[5,000, 40,000]` Hz range, `0.45·fs ≤ 18,000 Hz < 20,000 Hz`, so the
`20,000 Hz` term never actually binds today — `fc_max(fs)` reduces to `0.45·fs` for the
entire supported `fs` domain. It is kept as an explicit term because it is part of the
locked decision and guards against a future `fs` upper-bound increase silently raising
the effective cutoff past 20 kHz; it is not dead code to simplify away.

- **Invalid input → `ValueError`** raised from the constructor/setter, with a message
  naming the offending field and its valid range. Never silently clamp.
- UI layer catches `ValueError`, shows inline red validation text next to the field,
  and disables **Export** while *any* block in the chain is invalid.
- **`fs` changes are chain-wide and can invalidate existing blocks** (since `fc_max(fs)`
  depends on `fs`). Changing `fs` re-validates every block; any block whose current
  parameters (`fc`, or `f_low`/`f_high` for BP) now fall outside `[100, fc_max(fs)]` for
  the new `fs` gets an error badge. Values are **not** auto-clamped to the new range —
  the user must fix them explicitly.

---

## 6. Frequency-grid rules

Three distinct grids exist; they must not be conflated:

1. **Bode display grid** (per CONCEPT.md §4: "log scale, 10 Hz to
   `min(0.7·fs, fs/2−1)`"): `numpy.logspace(log10(10), log10(f_max), 500)`.
   **N = 500 is a proposed default**, not specified in CONCEPT.md — flagged as an open
   question (trade-off: curve smoothness vs. render cost on every parameter edit).
   **The `10 Hz` floor here is a display concern only, and is deliberately distinct from
   the `100 Hz` design-parameter minimum in §5:** the Bode plot always renders down to
   10 Hz for visual context — e.g. so the roll-off shape below a 100 Hz LP cutoff is
   still visible — even though no design parameter (`fc`, `f_low`, `f_high`) can ever be
   set below 100 Hz. Do not "fix" this by aligning the two numbers; the mismatch is
   intentional.
2. **Response-validation grid** (max/RMS amplitude error, per-block and combined):
   **identical to the Bode display grid** (500 log points, 10 Hz floor) — CONCEPT.md §6
   says "over the plotted frequency grid," which only makes sense if these are the same
   grid; kept explicit here to avoid an implicit second definition appearing later.
3. **Coefficient-accuracy sweep grid** (§11): `numpy.linspace(100, fc_max(fs), 1000)` —
   linear, not log, because it's a parameter sweep over `fc` (or BP's center frequency
   with bandwidth/Q held fixed), not a frequency axis. This sweeps the **design-parameter
   range** (§5), not the Bode display range — it uses the 100 Hz floor and `fc_max(fs)`
   ceiling, never the 10 Hz display floor. **Deviation from CONCEPT.md §6**, which
   describes this sweep as running "10 Hz to `0.45·fs`" — see Summary of deviations,
   item 6.
4. **Linear gain-plot grid** (`error_analysis.linear_response_grid()`): the same range as
   the Bode display grid (`BODE_FLOOR_HZ` to `min(0.7·fs, fs/2−1)`), but linearly, not
   logarithmically, spaced (`numpy.linspace`, not `numpy.logspace`). Feeds the linear
   frequency / linear gain plot that sits below the Bode plot in every Inspector tab
   (§13); gain itself is `10**(magnitude_db/20)`, converted at the plotting boundary
   from the same `ideal_coefficients()`/`q14_coefficients()` path as everywhere else —
   never a separately reimplemented filter equation.

All `freqz` calls take angular frequency `w = 2π·f/fs` (radians/sample) via
`worN=w_array` — never rely on scipy's own `worN=N` grid generation, since it wouldn't
match across the ideal/Q14 comparison or the fixed display grid.

---

## 7. Q14 rounding, overflow, and processing semantics

- **Scale factor:** 16384 (2¹⁴). This is *not* a strict Qm.n fractional format — LP/HP/BP/AP
  coefficients can reach magnitude ~2.0 (e.g. LP/HP `a1 → ±2` as `fc` approaches the
  band edges of its valid range), so at least 17 bits are needed to hold the scaled
  value. **Storage/ABI type is `int32_t`** for every coefficient (matches CONCEPT.md's
  header comment style, e.g. `int32_t` / `−22253`), which comfortably covers the full
  valid parameter domain — no int32 overflow is reachable for any `fc`/`f_low`/`f_high`/`Q`
  inside the ranges in §5. (A downstream firmware target that narrows to `int16_t` for
  its actual biquad state is a concern for the *consumer* of the generated header, not
  for this suite's own Q14 conversion or tests.)
- **Rounding mode:** round-half-away-from-zero, matching C's `roundf()`
  (`(int32_t)roundf(x * 16384.0f)`), **not** `numpy.round`'s round-half-to-even —
  those differ on exact ties and would silently poison the coefficient-accuracy sweep
  with spurious 1-LSB "errors" that are a rounding-mode mismatch, not real quantization
  error.
- **Architectural rule to prevent that class of bug entirely:** Python **never**
  reimplements Q14 quantization. Every `Q14Coefficients` value is produced by calling
  the compiled C library through ctypes (§8) — there is exactly one quantization
  implementation in the whole system, in C, and Python only ever reads its output back.
- **Overflow policy:** since int32 storage never overflows within the valid parameter
  domain, this is a defensive-only concern. The C functions should saturate to
  `INT32_MIN`/`INT32_MAX` rather than wrap on a hypothetical out-of-domain input — this
  is a "should never trigger" assertion-style guard, not an expected code path (Python
  validates parameters *before* calling into C; see §5, §8).
- **Signal processing (biquad_q14.c's `process()`) is out of scope for this app's own
  validation pipeline** — response comparison works entirely in the coefficient/transfer-
  function domain (dequantize → `freqz`, per §4), matching CONCEPT.md's literal wording
  ("reconstruct its transfer function"), not by running a literal fixed-point sample
  loop. `biquad_q14.c` exists as reference firmware source bundled into the deliverable
  (it's what actually runs on the Cortex-M4), not as something this desktop app executes
  against a live signal — there is no real-time audio preview (explicitly out of scope
  in CONCEPT.md §9).

**Automated test requirement (resolved):** although `process()` is out of scope for the
*app's own* validation pipeline (above), `biquad_q14.c` still ships as firmware-facing
reference source, and none of the other tests in this suite ever exercise its
delay-line/state-update logic — the response comparison in §4 stays entirely in the
coefficient/`freqz` domain and never calls `process()`. A bug confined to `process()`
(e.g. swapped delay registers, an off-by-one in the state shift, a saturation bug) would
therefore pass every other test in the suite undetected. To close that gap:

- `biquad_q14.c` **requires an automated impulse-response test**, distinct from the
  coefficient/response tests elsewhere in this document. For a representative set of
  designs (at minimum one LP, one HP, one BP, one AP, each at a few points spanning
  `[100, fc_max(fs)]`), feed a unit impulse through `process()` sample-by-sample and
  compare the resulting impulse response against a coefficient-domain reference (e.g.
  `scipy.signal.dimpulse` on the dequantized Q14 coefficients).
- This may be driven from Python via ctypes (calling `process()` directly, not through
  the design functions' coefficient output) or written as a standalone C test — either
  satisfies the requirement, as long as it actually invokes `process()`'s state-update
  path rather than only checking coefficients.
- Tolerance for this comparison is calibrated and locked alongside the other numerical
  tolerances in §11, once `biquad_q14.c` exists (Phase 2) — see §14.

---

## 8. Native C function signatures and ctypes ABI

`src/c/filter_design.h`:
```c
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t b0, b1, b2, a1, a2;   /* Q14 fixed-point, scale = 16384 */
} q14_coeffs_t;

/* Return 0 on success, negative on invalid input (defense-in-depth only —
   Python validates first per §5 and should never trigger these in practice). */
int filter_design_lp(float fc, float fs, q14_coeffs_t *out);
int filter_design_hp(float fc, float fs, q14_coeffs_t *out);
int filter_design_bp(float f_low, float f_high, float fs, q14_coeffs_t *out);
int filter_design_ap(float fc, float fs, float q, q14_coeffs_t *out);

#ifdef __cplusplus
}
#endif
```
Error codes: `0` success, `-1` frequency out of `(0, fs/2)`, `-2` `fs <= 0`,
`-3` `f_low >= f_high` (BP only), `-4` `q` out of `(0, ∞)` (AP only).

ctypes wrapper (`c_codegen.py`):
```python
class Q14Coeffs(ctypes.Structure):
    _fields_ = [("b0", ctypes.c_int32), ("b1", ctypes.c_int32),
                ("b2", ctypes.c_int32), ("a1", ctypes.c_int32), ("a2", ctypes.c_int32)]

lib.filter_design_lp.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.POINTER(Q14Coeffs)]
lib.filter_design_lp.restype  = ctypes.c_int
# filter_design_hp: same signature as lp
# filter_design_bp.argtypes = [c_float, c_float, c_float, POINTER(Q14Coeffs)]  # f_low, f_high, fs
# filter_design_ap.argtypes = [c_float, c_float, c_float, POINTER(Q14Coeffs)]  # fc, fs, q
```
- A nonzero C return code raises `RuntimeError` in the Python wrapper — this should be
  unreachable in normal operation since Python validates first; if it *does* trigger,
  treat it as a bug in the Python-side validation, not a normal error path.
- **All C functions must be pure/stateless** (no global/static mutable state) — ctypes
  calls happen synchronously on the Qt main thread, and statelessness is what makes
  that safe without locking.
- `extern "C"` guard included defensively even though the build is C, not C++.

---

## 9. Shared-library compilation and cache behavior

- **Compiler discovery**, in order: `$CC` env var (if set) → `gcc` → `cc` → `clang`.
- **Windows constraint (important, not obvious from CONCEPT.md):** CONCEPT.md says
  "GCC or Clang" and explicitly excludes MSVC. On Windows this means a MinGW-w64-style
  GCC, or Clang invoked in GNU-compatible mode (`clang --target=x86_64-w64-mingw32`),
  **not** `clang-cl` (MSVC-compatible mode) — ctypes needs a standard cdecl DLL without
  an MSVC-runtime dependency assumption. This distinction must be documented for anyone
  setting up a Windows dev/CI machine.
- **Compile command** (Linux): `gcc -O2 -fPIC -shared -o filter_design.so
  src/c/filter_design.c src/c/biquad_q14.c -Isrc/c -lm`
  (Windows/MinGW): `gcc -O2 -shared -o filter_design.dll src/c/filter_design.c
  src/c/biquad_q14.c -Isrc/c -lm` (no `-fPIC` needed).
- **Cache behavior — literal reading of CONCEPT.md's "compiled once at app start,
  cached until the next launch":** this means recompiled **every process launch**, into
  a temp/build directory, reused only for that process's lifetime — **not** cached to
  disk across separate app launches. This avoids stale-binary bugs if `src/c/*.c` is
  edited between runs, at the cost of a startup compile (~sub-second for two small
  files). No source-hash-based skip-if-unchanged logic in v1 — unconditional recompile
  keeps the behavior simple and impossible to get stale.
- **Failure handling:** compilation happens before the main window is shown. On
  failure, show a blocking modal with the compiler's stderr (first ~20 lines) and
  **Retry** / **Quit** buttons — the app cannot run without the native library (Q14 is
  load-bearing everywhere: Inspector, Export), so there is no degraded/mock mode in v1.
- Compilation is a single startup-only event — no in-app C source editing exists in v1,
  so there is no runtime recompilation trigger to design for.

---

## 10. Generated C-header format

**Revision from CONCEPT.md:** CONCEPT.md's template (`#define FILT1_B0 Q14(0.10234f)
// 1677`) leaves `Q14(...)` undefined — as written it would not compile standalone
unless the firmware consumer separately defines that macro, which conflicts with the
"self-contained deliverable... ready to drop into firmware" goal (CONCEPT.md §1).
**Resolution: emit plain integer literals as the `#define` value**, with the source
float and a note in a trailing comment for human readability:

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

- File name: `filter_design.h`, `#pragma once` guard.
- Filter numbering: `FILT<n>`, 1-indexed by chain position (left-to-right); renumbered
  on every reorder/add/delete so numbering is always contiguous `1..N`.
- Timestamp: ISO 8601 **with explicit UTC offset** (`datetime.now().astimezone().isoformat()`),
  not a bare local time — avoids ambiguity when a header generated on one machine is
  read on another.
- Coefficient order in the header matches §2: `B0, B1, B2, A1, A2`.
- ASCII, LF line endings, even when generated on Windows (firmware-side diffability).

**Resolved (release hardening): the export folder does not bundle
`biquad_q14.h`/`biquad_q14.c`.** CONCEPT.md §7's export directory listing
enumerates exactly `report.pdf`, `filter_design.h`, `bode_combined.png`,
`bode_<type>_<n>.png`, and `error_sweep_<n>.png` -- no `biquad_q14.*` entry,
and `filter_design.h` itself is specified above as coefficient-only.
`biquad_q14.{h,c}` remains firmware reference source living once in this
repository's `src/c/` tree (§7's "bundled into the deliverable" refers to
the suite's own source tree, not a per-export copy) -- duplicating it into
every timestamped export folder would create N copies with no benefit. A
firmware integrator combines the generated header with `src/c/biquad_q14.{h,c}`
themselves; that combination is proven, not just asserted, by
`tests/test_firmware_harness.py`, which compiles a harness `#include`-ing a
freshly generated header together with `biquad_q14.h`, links against
`biquad_q14.c` with `gcc -Wall -Wextra -Werror`, runs the resulting binary,
and cross-checks its output against the ctypes `process_impulse()` path
used elsewhere in this suite. See README.md §6.

---

## 11. Response and coefficient tolerances

- **Response pass/fail (user-facing):** max amplitude error `> 0.1 dB` over the display
  grid (§6) ⇒ warning badge on that block (and independently on the "Combined" cascade
  tab). Confirmed as-is from CONCEPT.md.
- **RMS is informational only** — no separate pass/fail gate; CONCEPT.md only ties the
  badge to max error.
- **Coefficient-accuracy sweep (§4) has no pass/fail threshold at all** — explicitly
  diagnostic per CONCEPT.md §6. Reported: max absolute error, RMS absolute error,
  worst-case sweep frequency, and per-coefficient maxima (`b0,b1,b2,a1,a2` each
  reported separately).
- **Automated test tolerances (pytest, distinct from the UI's 0.1 dB badge) — calibrated
  and locked in Phase 2** (`tests/test_native_coefficients.py`,
  `tests/test_native_impulse.py`):
  - **Coefficient error `< 1e-4` absolute**, per coefficient (`b0,b1,b2,a1,a2`), across
    the full fs sweep and a 1000-point design-parameter sweep. Measured error stays
    around `3e-5` in practice; `1e-4` keeps roughly 3x margin.
  - **Response (amplitude, dB) error is only evaluated where the ideal magnitude is
    above −20 dB.** Deep in the stopband the *ideal* float64 response reaches
    −150…−210 dB, but a 14-bit-quantized biquad has a real coefficient-quantization
    noise floor around −70…−100 dB — comparing dB error at those points measures "how
    deep can 14 bits of coefficient precision null a stopband" (a genuine, physically
    meaningful fixed-point limitation), not a defect. Left unguarded, individual points
    show >100 dB of "error" purely from comparing two near-zero numbers in log space, so
    every deep-stopband dB comparison is excluded by this magnitude gate — it is a
    property of Q14 quantization, not something to chase away by loosening the
    tolerance.
    - **Interior response error `< 0.5 dB`** (excluding the outer 5% of the design
      range, `[100, fc_max(fs)]`, at each end) — measured max ~0.25 dB (HP).
    - **Full-range response error `< 5.0 dB`** (including the extreme edges) — measured
      max ~3.65 dB (HP, `fc = 220 Hz` at `fs = 40,000 Hz`, the most quantization-sensitive
      corner of the design range).
  - **Impulse-response error `< 0.05`** (Q14-scale float units), comparing
    `biquad_q14_process()`'s sample-by-sample output against `scipy.signal.dimpulse()` on
    the dequantized Q14 coefficients, over **64 samples** (chosen because a lightly-damped
    2nd-order design near the low end of the supported `fc` range has a very slow decay —
    pole radius close to 1 — so per-sample drift from quantization-perturbed pole angle
    keeps growing with `n`; 64 samples captures the meaningful early transient without
    running into that unbounded long-tail drift). Measured max ~0.0198 (AP, `fc = 100 Hz`
    at `fs = 40,000 Hz`); `0.05` keeps roughly 2.5x margin.

  These values are fixed regression thresholds, not starting points — re-calibrate only
  if `filter_design.c` or `biquad_q14.c` change in a way that could shift the noise
  floor.

---

## 12. Chain-model ownership and cascade behavior

CONCEPT.md's file tree has no dedicated chain-model file — canvas.py implicitly owns
chain state as a UI concern. **Proposed addition:** `src/python/filters/chain.py`
holding a `FilterChain` model, decoupled from Qt, so it's unit-testable without a
running UI:

```python
class FilterChain:
    fs: float
    blocks: list[FilterDesign]          # ordered, index = display/export order

    def add_block(self, kind: str, **params) -> int: ...      # returns index
    def remove_block(self, index: int) -> None: ...
    def move_block(self, from_index: int, to_index: int) -> None: ...
    def combined_ideal_response(self, freq_hz: np.ndarray) -> FrequencyResponse: ...
    def combined_q14_response(self, freq_hz: np.ndarray, backend) -> FrequencyResponse: ...
```

- **Ownership rule:** `FilterBlockWidget` instances hold an *index* into
  `FilterChain.blocks`, never a private copy of filter parameters. `canvas.py` mutates
  the model, then re-renders widgets from it — single source of truth, prevents
  visual/data desync.
- **Cascade math:** series-only topology (no summing junction, per CONCEPT.md §9), so
  `H_combined(f) = Π H_i(f)`. In dB, magnitudes **add** (`Σ dB_i`); phases **add**, then
  unwrap for display.
- **`fs` is chain-wide**, not per-block (CONCEPT.md §4: shared input). Any `fs` edit
  propagates to every block and triggers the re-validation described in §5.
- **Empty chain:** `combined_*_response` on 0 blocks — Combined tab shows an explicit
  empty-state message, not a degenerate flat 0 dB / 0° line.
- Reordering/deleting renumbers `FILT<n>` labels (§10) and Inspector tabs immediately;
  deleting the currently-selected block falls back to selecting "Combined."

---

## 13. UI state, reset, dirty-state warning, and compiler-error handling

- **No persistence in v1** (save/load explicitly out of scope, CONCEPT.md §9) — all
  state is in-memory for the process lifetime.
- **"Clear" button** (canvas toolbar, per CONCEPT.md's mockup) empties the block list
  only; **`fs` is left unchanged** — it's a session-wide setting, not chain content,
  and resetting it unexpectedly on "Clear" would be a surprising UX. Destructive and
  irreversible (no undo, per CONCEPT.md §9) — must show a confirmation dialog
  ("Clear all N filter blocks?") whenever the chain is non-empty.
- **Validation is fully automatic — there is no Validate action anywhere in the UI.**
  `Inspector.refresh_validation()` re-runs the response-error / coefficient-sweep
  metrics (§11) after every model mutation: a parameter edit (on the field's
  `editingFinished`, not on every keystroke), a filter-type/add/remove/reorder change,
  a clear/reset, or an `fs` change. It then marks the chain clean, so metrics are never
  shown in a stale/greyed-out state — there is nothing to invalidate them against, since
  the recompute already happened. Live-recomputed views (Bode plot, the linear
  frequency/linear gain plot below it — see §6 item 4 — coefficient table) were already
  cheap and always current before this change and remain so. Invalid-block error
  messages and "backend/chain unavailable" states are unaffected by this and still
  displayed exactly as before.
- **The shared hover measurement cursor** (`ui/widgets/measurement_cursor.py`) is scoped
  to a single Inspector tab: it synchronizes a vertical frequency line across that tab's
  Bode and linear-gain plots on mouse movement alone (`motion_notify_event`, no click
  required), showing frequency plus Ideal/Q14 magnitude (dB), phase, and linear gain at
  the hovered point (via `numpy.interp` over the plotted response arrays). Each tab owns
  its own cursor instance — moving the pointer in one tab never moves another tab's
  cursor, and switching tabs never carries a cursor position over.
- **Export always forces a fresh, independent validation pass first**, regardless of
  the Inspector's own state — the shipped PDF's "Test results" section (CONCEPT.md §7)
  must never reflect a stale or out-of-band pass/fail number (e.g. from a chain mutation
  that bypassed the Inspector's own edit path). Export itself is not blocked by an
  existing failing badge (only by *invalid* parameters, §5) — a design that fails the
  0.1 dB check can still be exported; the report simply documents the failure.
- **Compiler-error handling** is a startup-only concern (§9): blocking modal with
  stderr excerpt, **Retry**/**Quit**. Since there's no in-app C editing, this is the
  only place a compiler failure can occur.

---

## 14. Module ownership, dependency order, and phase tests

*(Condensed here for completeness; full version also given in the end-of-task report.)*

| Phase | Module(s) | Depends on | Tests |
|---|---|---|---|
| 1 | `filters/base.py`, `lowpass.py`, `highpass.py`, `bandpass.py`, `allpass.py` | numpy, scipy only | Unit tests reproducing the §3 numeric verification (exact −3.0103 dB / phase points) for each type, across the full `[100, fc_max(fs)]` domain |
| 2 | `src/c/filter_design.{h,c}`, `src/c/biquad_q14.{h,c}`, `c_codegen.py` | Phase 1 (shares the §3 formulas) | Compile smoke test on Linux + Windows; ctypes round-trip test; coefficient-accuracy sweep (§11) to calibrate real tolerance constants; **automated impulse-response test for `biquad_q14.c`'s `process()` (§7)**, calibrated and locked alongside the other Phase 2 tolerances |
| 3 | `error_analysis.py` | Phases 1–2 | Response-error and coefficient-sweep correctness against known-good fixtures |
| 4 | `filters/chain.py` | Phase 1 | Cascade math (dB-additive), empty-chain, reorder/renumber |
| 5 | `ui/app.py`, `palette.py`, `canvas.py`, `inspector.py`, `widgets/*` | Phases 1–4 | Manual/UI smoke only in this phase; no pytest coverage expected for Qt widgets themselves |
| 6 | `export.py` | Phases 1–5 | Header text-format golden test; PDF generation smoke test; forced-fresh-validate-before-export behavior |

---

## Summary of deviations from CONCEPT.md

1. **BP formula replaced** — CONCEPT.md's single-K/linear-bandwidth pseudocode
   (undefined `Kw` term) does not hit exact −3 dB edges for non-narrow bands; replaced
   with a dual-edge-prewarped formula that is exact for any bandwidth (§3).
2. **AP formula made explicit** — CONCEPT.md described only the mirror-numerator
   structure; the `w0`/`alpha`/`Q` denominator formula is now specified (§3).
3. **"scipy-based ideal" clarified** — means "closed-form §3 equations evaluated in
   float64," not `scipy.signal.butter()` (§4).
4. **Header `Q14(...)` macro replaced with plain integer literals** so generated
   headers compile standalone (§10).
5. **"Cached until next launch" interpreted literally** as "recompiled every process
   start," not "cached to disk across launches" (§9).
6. **`fs` and cutoff ranges finalized, replacing CONCEPT.md's implicit ranges** — `fs` is
   now `[5,000, 40,000]` Hz (CONCEPT.md never stated a bound); the design-parameter
   cutoff range is now `[100, fc_max(fs)]` with `fc_max(fs) = min(20,000, 0.45·fs)`,
   replacing CONCEPT.md §6's "10 Hz to `0.45·fs`" sweep-range language (§5, §6). The
   Bode display's `10 Hz` floor is unchanged and intentionally kept separate from the
   new `100 Hz` design-parameter floor (§6).
7. **`biquad_q14.c` now has a required automated test** — an impulse-response
   cross-check of `process()` — resolving what was previously an open question (§7).
