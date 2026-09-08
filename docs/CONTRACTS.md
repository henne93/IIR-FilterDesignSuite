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

Fifteen contract areas, each with: the resolved decision, the reasoning, and (where
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
    kind: ClassVar[Literal["LP", "HP", "BP", "AP", "PK"]]

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

Concrete subclasses: `LowPassFilter`, `HighPassFilter`, `BandPassFilter`, `AllPassFilter`,
`PeakFilter` in `lowpass.py` / `highpass.py` / `bandpass.py` / `allpass.py` / `peak.py`, each implementing
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

### Peak (peaking EQ) — new filter type, not part of the original CONCEPT.md scope
Not a Butterworth design (unlike LP/HP/BP/AP) — a parametric bell boost/cut, derived from
the analog peaking-EQ prototype `H(s) = (s² + (A/Q)s + 1) / (s² + s/(A·Q) + 1)`,
`A = 10^(gain_dB/40)`, normalized to Ω₀=1. Bilinear-transformed via
`s = (1/K)·(1−z⁻¹)/(1+z⁻¹)`, `K = tan(π·fc/fs)` — the same prewarping LP/HP/BP use to
place the prototype's normalized critical frequency (Ω=1) exactly at the digital `fc`
(unlike All-Pass's un-prewarped `w0`):
```
K    = tan(π·fc/fs)
A    = 10^(gain_dB/40)
norm = K² + K/(A·Q) + 1
b0   = (K² + (A/Q)·K + 1) / norm
b1   = 2·(K² − 1) / norm
b2   = (K² − (A/Q)·K + 1) / norm
a1   = b1
a2   = (K² − K/(A·Q) + 1) / norm
```
`a1` and `b1` are identical by construction: the bilinear-transformed numerator and
denominator share the same `z⁻¹` coefficient (`2·(K²−1)`) before normalization by `norm` —
only the `z⁰`/`z⁻²` terms differ (they carry `±(A/Q)·K` vs. `±K/(A·Q)`).
Property: `|H(e^{j2π·fc/fs})|` = exactly `gain_dB` dB (pre-warping preserves the critical
point exactly, same reasoning as LP/HP's exact `-3.0103 dB` edge). `|H(DC)| = |H(Nyquist)|`
= exactly 0 dB (the prototype's `s→0`/`s→∞` boundary conditions both reduce to `H=1`,
independent of `Q` or `gain_dB`). At `gain_dB=0` (`A=1`), `b0=1` and `b1=a1`, `b2=a2`,
so `H(z)≡1` identically — a useful identity check.

**Verification method:** all four Butterworth formulas above were checked with a standalone
complex-arithmetic evaluation of `H(e^{jω})` (not scipy) at the claimed critical
frequencies, confirming `−3.0103 dB` (i.e. exact `−3 dB`) at every claimed edge and
exact `±0°`/`−180°` phase properties. Peak's exact-`gain_dB`-at-`fc` and exact-0dB-at-DC/
Nyquist properties were verified the same way (direct complex-arithmetic evaluation, not
scipy) — see `tests/test_peak.py`. Anyone re-implementing this in C should hold the
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
| `fc` (PK) | `100 Hz ≤ fc ≤ fc_max(fs)` | Same effective range as LP/HP/AP. |
| `Q` (PK) | `0.8 ≤ Q ≤ 4.0` | **Narrower than AP's `[0.25, 4.0]`** -- deliberate, see note below. Default `1.0`. |
| `gain` (PK) | `-15.0 dB ≤ gain ≤ 15.0 dB` | Product decision; default `+6.0 dB`. |

**Why PK's `Q` floor is 0.8, not AP's 0.25:** unlike LP/HP/BP/AP, whose coefficients are
provably bounded within ~2.0 in magnitude across their entire supported domain (§7), PK's
peaking-EQ coefficients are **not** bounded that way at low `Q` combined with high
`|gain|` -- e.g. `Q=0.25` at `gain=+15 dB` pushes `b0` up to ~3.11, which is real int16 Q14
saturation, not the ordinary quantization noise every other filter type exhibits. This was
discovered empirically while implementing PK (a domain sweep found `max_abs=32768`, i.e.
actual clamping, before this floor was raised) and resolved by raising `Q_MIN` to `0.8`,
which keeps the full `[-15, 15]` dB gain range while bringing the worst-case in-domain
coefficient back to `~1.998` (`|b1|`/`|a1|`, near `fc=100 Hz`, `Q=4.0`, `gain=±15 dB`) --
the same tight-but-non-saturating margin class as AP's own worst case. This is a
**deliberate, user-confirmed trade-off** (narrower low-Q range in exchange for keeping the
full gain range), not an oversight to "fix" by reverting to `0.25` -- see
`tests/test_native_coefficients.py::test_no_coefficient_saturates_int16_across_domain`,
which is the regression guard for this.

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
  coefficients can reach magnitude up to 2.0 in the theoretical Q14 format, but a full
  sweep of every LP/HP/BP/AP design across the actual supported parameter domain (§5:
  `fs∈[5000,40000]`, `fc`/`f_low`/`f_high`∈`[100,fc_max(fs)]`, `Q∈[0.25,4]`) gives a true
  worst-case magnitude of **1.9958** (AP `a1`/`b1`, near `Q=4`, `fc→100`). PK's own domain
  (`fc`/`fs` as above, `Q∈[0.8,4]` — narrower than AP's, see §5) gives a worst case of
  **~1.998** (`b1`/`a1`, near `fc→100`, `Q=4`, `gain=±15 dB`) — PK's coefficients are *not*
  bounded by ~2.0 the way LP/HP/BP/AP's are at lower `Q`, which is exactly why its `Q`
  floor is raised (§5). **Storage/ABI type is `int16_t`** for every coefficient — this
  fits inside int16 Q14 (max representable ≈1.99994) with only ~0.002–0.004 headroom
  depending on filter type, tight but real, and locked by an automated domain-sweep
  regression test (`tests/test_native_coefficients.py::test_no_coefficient_saturates_int16_across_domain`).
  This is narrower than an earlier int32-storage design (which had large headroom by
  construction) — the narrower width was chosen deliberately to keep the firmware-facing
  `biquad_q14.{h,c}` 16-bit-only end to end (coefficients, samples, and state), matching a
  16-bit ADC/DAC sample pipeline; see the "Signal processing" bullet below for the
  accumulator consequence of that choice.
- **Rounding mode:** round-half-away-from-zero, matching C's `roundf()`
  (`(int16_t)roundf(x * 16384.0f)`), **not** `numpy.round`'s round-half-to-even —
  those differ on exact ties and would silently poison the coefficient-accuracy sweep
  with spurious 1-LSB "errors" that are a rounding-mode mismatch, not real quantization
  error.
- **Architectural rule to prevent that class of bug entirely:** Python **never**
  reimplements Q14 quantization. Every `Q14Coefficients` value is produced by calling
  the compiled C library through ctypes (§8) — there is exactly one quantization
  implementation in the whole system, in C, and Python only ever reads its output back.
- **Overflow policy (coefficients):** the int16 headroom above is tight but never
  actually reached in-domain (regression-guarded, see above) — the C functions still
  saturate to `INT16_MIN`/`INT16_MAX` rather than wrap on a hypothetical out-of-domain
  input, as a defensive guard (Python validates parameters *before* calling into C; see
  §5, §8).
- **Overflow policy (`biquad_q14_process`'s accumulator) — accepted trade-off, not a
  proven-safe design:** samples/state (`x1,x2,y1,y2`, the `process()` argument/return)
  are also `int16_t`. Each of the 5 coefficient×sample products is computed as `int32_t`
  (safe: int16×int16 has magnitude at most ~2³⁰). Summing all 5, however, has a
  theoretical worst case around 5.4×10⁹ (using the full Q14 coefficient range) — about
  2.5× over `int32_t`'s range — so a plain `int32_t` sum would risk genuine
  signed-integer-overflow UB. The recommended, fully-safe option was a widened
  `int64_t` accumulator (as the original design used, and as ARM's own CMSIS-DSP
  `arm_biquad_cascade_df1_q15` does for exactly this reason — it costs nothing on
  Cortex-M4, which has single-cycle 32×32→64-bit MAC hardware). **The user was told this
  and explicitly chose a strict 32-bit-accumulator, zero-64-bit-types design instead.**
  `biquad_q14.c` implements this with explicit saturating add/subtract
  (`__builtin_add_overflow`/`__builtin_sub_overflow`, GCC/Clang-only, matching §9's
  compiler restriction) at every one of the 5 accumulation steps plus the Q28→Q14
  rescale, clamping to `[INT32_MIN, INT32_MAX]` instead of wrapping. This guarantees
  *well-defined* behavior (no UB, no crash) but is **not a formal proof that saturation
  is unreachable** for every in-domain design — no per-filter-type pole/gain stability
  bound has been derived. `tests/test_native_saturation_stress.py` empirically checks
  this: a bit-exact Python mirror of the saturating algorithm is compared against the
  real compiled `process()` under adversarial full-scale alternating input for
  representative designs (including the domain's tightest AP corner), reporting how
  often — if ever — the clamp is actually reached.
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
- **Additional requirement (16-bit/32-bit-accumulator design):** since the 32-bit
  accumulator's overflow-freedom is an accepted trade-off rather than a proof (see
  above), `biquad_q14.c` also requires the saturation-stress test described above
  (`tests/test_native_saturation_stress.py`) — a bit-exact comparison against a Python
  mirror of the saturating algorithm under adversarial input, reporting observed
  saturation-event counts. This is a permanent companion to the impulse-response test,
  not a one-time calibration.

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
    int16_t b0, b1, b2, a1, a2;   /* Q14 fixed-point, scale = 16384 */
} q14_coeffs_t;

/* Return 0 on success, negative on invalid input (defense-in-depth only —
   Python validates first per §5 and should never trigger these in practice). */
int filter_design_lp(float fc, float fs, q14_coeffs_t *out);
int filter_design_hp(float fc, float fs, q14_coeffs_t *out);
int filter_design_bp(float f_low, float f_high, float fs, q14_coeffs_t *out);
int filter_design_ap(float fc, float fs, float q, q14_coeffs_t *out);
int filter_design_pk(float fc, float fs, float q, float gain_db, q14_coeffs_t *out);

#ifdef __cplusplus
}
#endif
```
Error codes: `0` success, `-1` frequency out of `(0, fs/2)`, `-2` `fs <= 0`,
`-3` `f_low >= f_high` (BP only), `-4` `q` out of `(0, ∞)` (AP, PK).

ctypes wrapper (`c_codegen.py`):
```python
class Q14Coeffs(ctypes.Structure):
    _fields_ = [("b0", ctypes.c_int16), ("b1", ctypes.c_int16),
                ("b2", ctypes.c_int16), ("a1", ctypes.c_int16), ("a2", ctypes.c_int16)]

lib.filter_design_lp.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.POINTER(Q14Coeffs)]
lib.filter_design_lp.restype  = ctypes.c_int
# filter_design_hp: same signature as lp
# filter_design_bp.argtypes = [c_float, c_float, c_float, POINTER(Q14Coeffs)]  # f_low, f_high, fs
# filter_design_ap.argtypes = [c_float, c_float, c_float, POINTER(Q14Coeffs)]  # fc, fs, q
# filter_design_pk.argtypes = [c_float, c_float, c_float, c_float, POINTER(Q14Coeffs)]  # fc, fs, q, gain_db

# biquad_q14_process (src/c/biquad_q14.h) is 16-bit-only end to end:
# lib.biquad_q14_process.argtypes = [ctypes.POINTER(BiquadState), ctypes.c_int16]
# lib.biquad_q14_process.restype  = ctypes.c_int16
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

- File name: `filter_design.h` (written to `source/biquad_q14/gen/filter_design.h`
  -- see below), `#pragma once` guard.
- Filter numbering: `FILT<n>`, 1-indexed by chain position (left-to-right); renumbered
  on every reorder/add/delete so numbering is always contiguous `1..N`.
- Timestamp: ISO 8601 **with explicit UTC offset** (`datetime.now().astimezone().isoformat()`),
  not a bare local time — avoids ambiguity when a header generated on one machine is
  read on another.
- Coefficient order in the header matches §2: `B0, B1, B2, A1, A2`.
- ASCII, LF line endings, even when generated on Windows (firmware-side diffability).

**Export directory layout (revised -- decision log entry 15 below):** every
export produces

```
export_YYYYMMDD_HHMMSS/
├── design.iirfilt                      current chain config, project_file.save_project()
├── source/                             self-contained C deliverable
│   ├── biquad_q14/
│   │   ├── cfg/                        reserved for future user-config macros (empty)
│   │   ├── inc/
│   │   │   ├── biquad_q14.h            repoints its filter_design.h include at filter_design_calc.h
│   │   │   └── filter_design_calc.h    verbatim copy of src/c/filter_design.h
│   │   ├── src/
│   │   │   ├── biquad_q14.c            verbatim copy of src/c/biquad_q14.c
│   │   │   └── filter_design_calc.c    verbatim copy of src/c/filter_design.c except one #include line
│   │   └── gen/
│   │       └── filter_design.h         GENERATED per-design Q14 coefficients (FILT<n>_* defines) --
│   │                                   the ONLY copy of this file anywhere in the export
│   ├── app_template/
│   │   └── example.c                   generated cascade-wiring demo
│   └── README.md                       integration instructions
└── reports/
    ├── biquad_q14_report.pdf           renamed from report.pdf; content/generation unchanged
    ├── figures/
    │   ├── bode_combined.png
    │   ├── bode_<kind>_<n>.png
    │   ├── error_sweep_<n>.png         (moved from the export root; no content change)
    │   ├── time_domain_combined.png    optional -- see "Time-domain export artifacts" below
    │   └── time_domain_<kind>_<n>.png  optional, same numbering as bode_<kind>_<n>.png
    ├── data/
    │   ├── time_domain_combined.csv    optional, same condition as the PNGs above
    │   └── time_domain_<kind>_<n>.csv  optional, same numbering
    └── test/
        └── test_summary.txt            C compile+run validation result (see below)
```

`design.iirfilt` is written unconditionally, every export, via
`project_file.save_project(chain, path, signal_chain)` -- a fixed filename
(`PROJECT_FILENAME` in `export.py`, built from
`project_file.PROJECT_FILE_EXTENSION` rather than hardcoding `.iirfilt` a
second time), independent of whatever project file the user separately has
open via File ▸ Save/Save As, so this export directory round-trips to exactly
the chain state that was exported. It captures the *full* `FilterChain`
(including disabled/invalid blocks, per §15), not just the active blocks the
rest of the export covers -- and, since the Time-Domain export addendum
below, the *full* `SignalChain` the same way (every signal block, valid or
not, per §15's schema v2 scope) when `export_design()` was given one;
omitting `signal_chain` (as `export.py`'s own internal call site did before
that addendum, and any caller with no `SignalChain` of its own still can)
writes a well-formed file with an empty `signals` list, never a missing key.

The old top-level, coefficient-only `filter_design.h` is gone: `source/biquad_q14/gen/filter_design.h`
is now the only copy of the generated header anywhere in the export. The
PDF's "C header listing" section (item 4 below) is unaffected -- it renders
`render_header(snapshot)` as text directly, with no file dependency.
`tests/test_firmware_harness.py` (renamed target path, same intent) still
proves this generated header combines correctly with this suite's own
`src/c/biquad_q14.{h,c}` reference source, `-I`-ing the new `gen/` directory.

**`source/` bundles a complete, self-contained C package** (unchanged intent
from the original `firmware/` subfolder, restructured -- see decision log
entries 10/13/14 below for that history). It contains: the same coefficient
header, generated fresh, at `source/biquad_q14/gen/filter_design.h`; the Q14
design-function implementation itself (`filter_design_lp/hp/bp/ap/pk()`),
copied verbatim from `src/c/filter_design.{h,c}` but renamed to
`filter_design_calc.{h,c}` -- the rename is the only change, still forced by
the naming collision with the generated coefficient header, which keeps the
`filter_design.h` name; a `biquad_q14.h` variant whose
`#include "filter_design.h"` line is repointed at the bundled
`filter_design_calc.h` (same collision, same fix) instead of pulling in the
generated coefficient header; a byte-for-byte verbatim copy of
`src/c/biquad_q14.c` (never hand-duplicated, so Feature-A-style changes to
the real implementation propagate to the next export automatically); a
generated `example.c` wiring up the *specific* chain being exported (one
`biquad_q14_state_t` per active block, chained in series through a
`process_chain()` function, matching §12's series-only topology); and a
`README.md` with integration instructions. `example.c` demonstrates the
bundled design functions rather than the frozen defines:
`filter_chain_init()` calls each block's `filter_design_lp/hp/bp/ap/pk()`
with that block's own exported design parameters as literal float args
(e.g. `filter_design_lp(3000.0, 13333.0, &coeffs1)`), not the `FILT<n>_*`
values from `filter_design.h`. `render_source_package()` (`export.py`,
renamed from `render_firmware_package()`) produces all of this from the same
`ExportSnapshot` used elsewhere, with no new data model. This is proven
fully self-contained -- not just asserted -- by `tests/test_source_package.py`
(renamed from `test_firmware_package.py`), which copies the generated
`source/` folder to a location with no relationship to this repository,
compiles it there with `-I` pointed only at that copy's `biquad_q14/inc`
(no reference to `src/c/` at all), runs the binary, and cross-checks its
output bit-exactly against the ctypes `NativeBackend.process_samples()` path
used elsewhere in this suite; a further test in that file compiles a small
harness calling `filter_design_lp/hp/bp/ap/pk()` directly from the detached
`filter_design_calc.{h,c}` pair and cross-checks the resulting Q14
coefficients bit-exactly against `NativeBackend.design_lp/hp/bp/ap/pk()` --
which is itself the same source, compiled unrenamed, and validated against
the Python/scipy "ideal" coefficients across the full parameter domain by
`tests/test_native_coefficients.py` (§4/§6/§11). See README.md §6.

**Internal layout of `source/` (revised -- decision log entry 15):**
`biquad_q14/` splits every filter *source* file across four subdirectories --
`cfg/` (reserved for future user-config macros, currently always empty),
`inc/` (both headers -- `biquad_q14.h` and the renamed `filter_design_calc.h`
-- as bare sibling names), `src/` (`biquad_q14.c` and `filter_design_calc.c`),
and `gen/` (the generated, per-design `filter_design.h`). `app_template/`
(holding the generated `example.c`) is a **sibling** of `biquad_q14/`, not
its parent -- unlike the old flat `firmware/` top level, where `example.c`
sat directly alongside the `biquad_q14/` subfolder it reached via a
subfolder-qualified include. Because `app_template/` and `biquad_q14/` are
now both children of `source/`, `example.c`'s own `#include`s are bare names
(`#include "filter_design_calc.h"`, `#include "biquad_q14.h"`) -- quoted-include
same-directory resolution alone would not find them -- and compiling it
requires an explicit `-I` flag pointed at `biquad_q14/inc`. The canonical
compile command (used consistently in `source/README.md`, in the C validation
step below, and in `tests/test_source_package.py`):

```
gcc -Wall -Wextra -Werror -std=c11 -I biquad_q14/inc app_template/example.c \
    biquad_q14/src/biquad_q14.c biquad_q14/src/filter_design_calc.c -o demo -lm
```

(cwd = `source/`; pass absolute/relative paths from elsewhere as needed.)

**C validation step (NEW -- decision log entry 15):** `export_design()`
(`export.py`) calls a new `run_c_validation(snapshot, source_dir,
reports_test_dir, backend)` after `render_source_package()` writes
`source/`, which compiles and runs that *same, freshly-written* package (not
a detached copy) with the canonical compile command above, then two
cross-checks: (1) a **cascade check** -- run `app_template/example.c`'s
compiled binary and compare its printed samples against
`backend.process_samples(block.q14_coefficients, impulse)` chained
stage-by-stage across `snapshot.blocks`, impulse = `[16384] + [0]*(N-1)`; (2)
a **design-function check** -- a small generated harness calls each active
block's own `filter_design_lp/hp/bp/ap/pk()` (its own exported params as
literal float args, the same statement-builder `example.c` itself uses) and
compares the printed Q14 coefficients directly against
`snapshot.blocks[i].q14_coefficients` (already computed via the same
`NativeBackend` at `build_snapshot()` time). The result -- an overall
PASS/FAIL/SKIPPED verdict plus per-check detail (compiler stderr on a
compile failure, the specific mismatched sample/coefficient on a numeric
failure) -- is written to `reports/test/test_summary.txt`.

This step is implemented as production code inside `export.py` rather than
shelling out to `pytest` at runtime, deliberately: `tests/test_source_package.py`'s
own proof already builds its fixture via `export_design()`, so invoking
`pytest` from inside `export_design()` would recurse, and a packaged/installed
build ships no `pytest` or `tests/`/`source/` tree to shell out to in the
first place. It is also, by hard requirement, **never allowed to fail
export**: matching this module's own "export is blocked only by invalid
parameters" philosophy (§13) taken one step further, `run_c_validation()`
catches gcc-missing (`FileNotFoundError`), any non-zero compile/run exit, a
30s-compile/10s-run timeout, and any numeric mismatch, recording each as
FAIL or SKIPPED in `test_summary.txt` instead of raising `ExportError` --
and a last-resort catch-all around the whole step guards against a bug in
the validation logic itself doing the same. `export_design()` still writes
every other file even if this step finds real problems (or can't run at
all, e.g. no `gcc` on `PATH`).

**Time-domain export artifacts (NEW -- reverses CONCEPT.md §11.8's original
"exporting time-domain plots/data" exclusion):** `export_design()` gained
three optional keyword parameters -- `signal_chain: SignalChain | None`,
`time_domain_duration_ms: float | None`, `time_domain_full_scale: int | None`
(all default `None`) -- wired from `ui/app.py`'s `_on_export()` to the
Time-Domain view's own `SignalChain` and its Inspector's *current*
`duration_ms()`/`full_scale()` field values (CONCEPT.md §11.4), not a fixed
export-only default: unlike the Bode sweep grid (`bode_grid(fs)`, unrelated
to any UI zoom/setting), duration and full-scale are the user's own
deliberate, already-exposed exploration settings, so export reflects exactly
what the Time-Domain Inspector currently shows.

For each of `Combined` + one per *active* filter block (same
`_active_blocks()`/`FILT<n>` numbering as `bode_<kind>_<n>.png`,
`time_domain.compute()` fed `chain.valid_filters` for Combined or
`[block.filter]` for a single block), this writes a PNG plot
(`reports/figures/time_domain_combined.png` /
`time_domain_<kind>_<n>.png`) and a CSV
(`reports/data/time_domain_combined.csv` / `time_domain_<kind>_<n>.csv`,
header row `time_ms,source,ideal,q14`, one row per sample, LF-only UTF-8
same as the generated header/project file). `q14` is never blank/omitted in
the CSV -- export always requires a `NativeBackend` (`build_snapshot()`
rejects a missing one before any of this runs) -- so all four columns are
always populated. `render_pdf()` also embeds each of these PNGs under a
"Time domain" sub-heading, immediately after that section's existing Bode
image (per-block sections and the Combined section alike).

This never blocks the rest of the export, mirroring §13's "invalid
parameters on an *enabled* block" carve-out one step further: omitting
`signal_chain` entirely (`export.py`'s own historical behavior, still the
default) skips all of this outright; an empty signal chain, one whose
blocks are all invalid, or a missing/non-positive
`time_domain_duration_ms`/`time_domain_full_scale` (e.g. the user's
duration/full-scale field was mid-edit and unparsable when Export was
triggered -- `ui/app.py` passes `None` for either rather than raising)
silently omits just the four `time_domain_*`/`data/*` artifacts and their
now-unneeded `reports/data/` directory, while the PDF, C `source/` package,
Bode/error-sweep plots, and `design.iirfilt` are written exactly as without
a signal chain -- never a `ValueError`, never an empty placeholder file. See
`export._time_domain_source()`.

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

- **Persistence: save/open project files** (reversing CONCEPT.md §9's original
  exclusion) — a `FilterChain`'s full state (`fs` + every block, valid or not, enabled or
  not) can be saved to and reloaded from a versioned JSON project file. See §15 for the
  format and the UI's in-place-mutation rule. UI chrome (splitter sizes, selected tab,
  window geometry) is never persisted — only chain-model state.
- **"Clear" action** (Edit menu — see below) empties the block list
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

## 15. Project file format (save/open)

Added after v1's initial "no persistence" decision (§13) was reversed.

- **Format:** versioned JSON, one object per file. Current (`schema_version: 2`,
  §11's signal chain added):
  ```json
  {
    "schema_version": 2,
    "fs": 13333.0,
    "blocks": [
      {"kind": "LP", "params": {"fc": 3000.0}, "enabled": true}
    ],
    "signals": [
      {"kind": "SIN", "params": {"frequency": 1000.0, "amplitude": 0.5, "phase_deg": 0.0}, "factor": 1.0},
      {"kind": "NOISE", "params": {"amplitude": 0.05, "seed": 123456}, "factor": 1.0}
    ]
  }
  ```
  `schema_version` is checked on load; an unrecognized value is rejected with an
  actionable error rather than guessed at. A `version 1` file (no `signals` key at
  all -- the format before §11's signal chain existed) still loads, with an empty
  signal chain; `save_project()` always writes the current version. File extension:
  `.iirfilt`. Written as LF-only UTF-8 (same technique as §10's generated header) so
  line endings never depend on platform.
- **Scope — model-only, no UI chrome:** a project file captures a `FilterChain`'s
  round-trip state — chain-wide `fs`, plus every block's `kind`, `params`, and
  `enabled` flag, **in chain order, including invalid and disabled blocks** — and,
  since schema v2, a `SignalChain`'s round-trip state the same way: every signal
  block's `kind`, `params`, and `factor`, **in chain order, including invalid
  blocks** (signal blocks have no `enabled` flag, §11.2 has none). A `NOISE`
  block's `seed` is just another entry of its own `params` dict (see
  `signals/chain.py`'s `add_block()`), so it round-trips with no dedicated field --
  this also fixes what would otherwise be a reproducibility gap (§11's noise curve
  changing on every reload). A `CSV` block's `file_path` is stored verbatim
  (whatever the file picker returned, normally absolute) -- never rewritten to be
  relative to the project file; if the path no longer resolves at load time the
  block still round-trips, just marked invalid with `.error` set exactly like any
  other invalid block (never a load-time failure). None of this is an export
  deliverable (unlike `export.py`'s `ExportSnapshot`, which is active-only and
  freshly validated). UI chrome (splitter sizes, selected block/tab, window
  geometry) is never persisted.
- **`src/python/project_file.py`** owns `save_project(chain, path, signal_chain=None)`
  / `load_project(path) -> (fs, blocks, signal_blocks)`. `signal_chain` is optional
  (`export.py`'s own call site has no `SignalChain` of its own and omits it, which
  still writes a well-formed current-schema file with an empty `signals` list, not
  a missing key). `load_project` returns plain data — it does **not** construct a
  `FilterChain`/`SignalChain` itself; the UI layer applies the result to the chains
  it already has.
- **UI rule — mutate the existing chains in place, never construct new ones:**
  `ui/app.py`'s Open handler follows exactly the same pattern as the existing Reset
  action (§13): if either chain is dirty, confirm via the same style of dialog; on
  proceeding, set `chain.fs` first (validates), then `chain.clear()`, then
  `chain.add_block(kind, **params)` per saved block (re-applying `enabled` via
  `chain.set_enabled()`), then `chain.mark_clean()` -- and, the same way, the
  `SignalChain`'s `clear()` + `add_block(kind, factor=factor, **params)` per saved
  signal block, then its own `mark_clean()`. `canvas`/`inspector` (both views) hold
  references to the original `FilterChain`/`SignalChain` instances, so both are
  mutated, never replaced. A malformed file is fully parsed and validated *before*
  any mutation begins, so a bad file can never leave the app in a half-applied
  state.
- **`SignalChain` has its own `dirty`/`mark_clean()`** (mirrors `FilterChain`'s,
  §13), set on every mutating method. The title-bar `"*"` and the
  unsaved-changes-on-close/-open warnings now check *either* chain's `dirty` flag
  (`MainWindow._is_dirty()`) -- since the signal chain is part of the project file
  as of schema v2, an unsaved signal-block edit must warn exactly like an unsaved
  filter-block edit.
- Save/Save As/Open live in the **File** menu, alongside Export (Clear/Reset
  live in the **Edit** menu) — reversing this section's original "no menu bar"
  assumption; there is no toolbar in this app. Save
  without a known path behaves like Save As. Saving marks both chains clean (same
  `dirty` flag that already drives the title-bar `"*"` and the unsaved-changes-on-close
  warning, §13) and remembers the path for a subsequent plain Save.

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
8. **`q14_coeffs_t` and `biquad_q14_state_t` narrowed from `int32_t` to `int16_t`,
   with the MAC accumulator forced to `int32_t` (saturating) instead of the originally
   safer `int64_t`** — a deliberate, user-directed trade-off after being shown the
   domain-sweep headroom numbers and the accumulator's residual (unproven-safe) overflow
   risk; see §7's "Overflow policy (`biquad_q14_process`'s accumulator)" bullet.
9. **Save/load implemented, reversing CONCEPT.md §9's original exclusion** — a
   versioned JSON project file (§15) captures a `FilterChain`'s full round-trip state;
   see §13's persistence bullet.
10. **Export's `firmware/` subfolder now bundles `biquad_q14.{h,c}` (as a generated,
    standalone-package variant, plus a generated cascade example)** — reversing, for
    this new subfolder only, the earlier "does not bundle" decision (§10); the
    top-level export files are unchanged.
11. **Toolbar replaced with a File/Edit menu bar** — File holds Open/Save/Save
    As/Export, Edit holds Clear/Reset. Purely a UI-chrome change; no behavioral
    contract above depends on which widget triggers an action (§13).
12. **Peak (PK), a fifth, non-Butterworth filter type, added** — parametric bell
    boost/cut; see §3, §5, §7 for its formula and the tighter `Q` range this
    required.
13. **Export's `firmware/` subfolder now also bundles the Q14 design-function
    implementation itself, as `filter_design_calc.{h,c}`, and the generated
    `example.c` now calls it** — reversing, for this subfolder only, the earlier
    "firmware doesn't need filter_design_lp/hp/bp/ap()" stance (§10); renamed from
    `filter_design.{h,c}` solely to avoid colliding with the generated coefficient
    header of the same original name already in the package. `example.c`'s
    `filter_chain_init()` now computes each stage's coefficients at runtime via
    these functions (each block's own design params as literal float args) instead
    of reading the frozen `FILT<n>_*` defines from `filter_design.h` directly --
    demonstrating the runtime-recompute path, e.g. for retuning a filter, while
    `filter_design.h`'s frozen defines remain available for a design that never
    changes after flashing.
14. **Export's `firmware/` subfolder now nests every filter *source* file under
    a `firmware/biquad_q14/` subdirectory** -- `biquad_q14.{h,c}` and
    `filter_design_calc.{h,c}` move down one level; `filter_design.h`,
    `example.c`, and `README.md` stay at `firmware/`'s own top level. Purely a
    layout change (§10): no file's *content* changed beyond `example.c`'s own
    `#include` paths gaining the `biquad_q14/` prefix they need to still find
    their headers.
15. **Export directory restructured into `design.iirfilt` / `source/` / `reports/`
    top-level groups; `source/`'s C package split into `cfg/inc/src/gen`; a real
    compile-time C validation step added** (§10) -- the export directory's flat
    file/PNG mix plus a `firmware/` subfolder is replaced by three clearly-scoped
    top-level entries: `design.iirfilt` (the exported chain's own round-trip
    project file, written via `project_file.save_project()` -- new; folding this
    in means an export always round-trips to exactly what was shipped, independent
    of whatever project file the user separately has open), `source/` (renamed
    from `firmware/`; the complete, self-contained C deliverable), and `reports/`
    (the PDF, now `biquad_q14_report.pdf`, plus `figures/` for every PNG and a new
    `test/` for the C validation summary). Within `source/`, `biquad_q14/` is
    split into `cfg/` (reserved, empty), `inc/` (headers), `src/` (implementation),
    and `gen/` (the generated coefficient header -- now the *only* copy anywhere in
    the export, replacing the old top-level/`firmware/` duplication), and
    `app_template/` (holding the generated `example.c`) becomes a sibling of
    `biquad_q14/` rather than sitting alongside it at a shared `firmware/` top
    level with a subfolder-qualified include path -- `example.c`'s `#include`s are
    now bare names, and compiling requires an explicit `-I biquad_q14/inc` flag
    (the canonical compile command is documented once, in `export.py`'s module
    docstring, `source/README.md`, and the C validation step, all consistently).
    New: `export_design()` now also compiles and runs the just-written `source/`
    package as a real, in-process production validation step
    (`run_c_validation()`) -- a cascade compile+run cross-check against
    `NativeBackend.process_samples()` and a design-function cross-check against
    each block's own `snapshot.q14_coefficients` -- recording a PASS/FAIL/SKIPPED
    verdict to `reports/test/test_summary.txt`. Implemented as production code
    rather than shelling out to `pytest` at runtime (which would recurse, since
    the dev-test proof for this same package already calls `export_design()`
    itself, and would break in a packaged build with no `pytest`/`tests`/`source`
    tree available) and, matching this module's own "export is blocked only by
    invalid parameters" philosophy (§13) taken one step further, engineered to
    NEVER fail or block export regardless of what it finds -- a missing compiler
    or a real compile/run/mismatch failure is recorded in the summary, not raised.
    Motivation throughout: organize the deliverable into clearly-scoped top-level
    groups (what a human reads vs. what a firmware integrator compiles vs. the
    round-trip project state), and replace an implicit "trust the generated files"
    posture with a real, automated, non-blocking compile-time proof that ships
    with every export.
16. **Signal-chain (§11) persistence added, reversing §11.8's original "TBD, likely
    a later iteration" exclusion** — the project file format bumps to
    `schema_version: 2`, adding a `signals` section that round-trips the
    `SignalChain`'s blocks (kind/params/factor, including invalid ones) the same
    way `blocks` already round-trips the `FilterChain`; a `NOISE` block's `seed`
    rides along inside its own `params` (no dedicated field needed), fixing what
    would otherwise be a reproducibility gap across reloads. A version-1 file (no
    `signals` key) still loads, with an empty signal chain. `SignalChain` gained
    its own `dirty`/`mark_clean()` so the title-bar `"*"` and unsaved-changes
    warnings cover signal-block edits too (§15).
17. **Noise reseed control added** — a `NOISE` block's seed was previously fixed at
    creation with no UI-visible way to change it (§11's stability guarantee: stays
    fixed across unrelated edits/reloads). `SignalChain.reseed(block_id)` draws a
    fresh random seed and re-validates via the existing `update_params()` path
    (raises `ValueError` for a non-`NOISE` block, `KeyError` for an unknown one);
    the Noise tile (`ui/widgets/signal_block.py`) now shows the current seed and a
    **Reseed** button wired to it (`SignalBlockWidget.reseed_requested` →
    `SignalCanvas.reseed_block()`), the one explicit, user-triggered exception to
    that stability guarantee.
18. **Time-domain export artifacts added, reversing §11.8's original "may follow
    later" exclusion** — `export_design()` gained optional `signal_chain`/
    `time_domain_duration_ms`/`time_domain_full_scale` parameters (§10's
    "Time-domain export artifacts" addendum); when a non-empty, at-least-partly-
    valid signal chain and a usable duration/full-scale are given, export writes
    a Combined + per-active-filter-block PNG (`reports/figures/time_domain_*.png`)
    and CSV (`reports/data/time_domain_*.csv`, `time_ms,source,ideal,q14`) pair,
    and `render_pdf()` embeds each PNG under a new "Time domain" sub-heading.
    Never blocks export: an omitted/empty/fully-invalid signal chain or an
    unusable duration/full-scale just omits these artifacts, matching §13's
    "invalid parameters on an *enabled* block" non-blocking philosophy. Also:
    `design.iirfilt` now round-trips the exported `SignalChain` too (via
    `save_project()`'s existing `signal_chain` parameter, previously unused by
    `export.py`), so an export's own project file matches exactly what was
    exported, signal chain included.
