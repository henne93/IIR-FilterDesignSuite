#pragma once
#include <stdint.h>

#include "filter_design.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generic Direct Form 1 Q14 biquad — reference firmware source
 * (CONTRACTS.md §7: bundled as deliverable firmware source, not executed by
 * this app's own validation pipeline, but required to carry an automated
 * impulse-response test).
 *
 * This struct layout and function signature are this suite's own design —
 * CONTRACTS.md §8 only contracts q14_coeffs_t and the filter_design_*
 * functions; the biquad process/state shape is left to the implementation.
 * Samples and state (x1, x2, y1, y2) are Q14 fixed-point, same scale
 * (Q14_SCALE = 16384) as q14_coeffs_t, e.g. a full-scale unit sample is
 * represented as 16384.
 *
 * All types here are int16_t/int32_t only -- no 64-bit type appears anywhere
 * in this file, by deliberate choice (see biquad_q14_process below for the
 * accumulator trade-off this implies).
 */
typedef struct {
    q14_coeffs_t coeffs;
    int16_t x1, x2;   /* x[n-1], x[n-2] */
    int16_t y1, y2;   /* y[n-1], y[n-2] */
} biquad_q14_state_t;

void biquad_q14_init(biquad_q14_state_t *state, const q14_coeffs_t *coeffs);

/*
 * Processes one Q14 input sample through
 *   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 * (Direct Form 1; minus signs on the a-terms match the CONTRACTS.md §2
 * denominator convention 1 + a1*z^-1 + a2*z^-2) and returns y[n].
 *
 * Each of the 5 coefficient*sample products is computed as int32_t (safe:
 * int16 x int16 has magnitude at most ~2^30, far inside int32_t range). The
 * 5 products are then summed via saturating add/subtract (see
 * sat_add_i32/sat_sub_i32 in biquad_q14.c), clamping to [INT32_MIN,
 * INT32_MAX] at every step instead of wrapping.
 *
 * This is a deliberate, informed trade-off: the theoretical worst-case
 * magnitude of the 5-term sum (~5.4e9, using the full Q14 coefficient range)
 * is about 2.5x over int32_t's range, so a plain `int32_t acc = ...+...;`
 * sum would risk genuine signed-integer-overflow UB. A 64-bit accumulator
 * would rule that out entirely (and costs nothing extra on Cortex-M4, which
 * has single-cycle 32x32->64-bit MAC hardware) -- that was the recommended
 * option, but a strict "no 64-bit types anywhere" requirement was chosen
 * instead. The saturating adds/subtracts above guarantee *well-defined*
 * behavior (no UB, no crash) but do NOT constitute a formal proof that
 * saturation is unreachable for every in-domain design; no such proof (a
 * per-filter-type pole/gain stability bound) has been derived. See
 * tests/test_native_saturation_stress.py for an empirical, bit-exact check
 * of this accumulator against a Python reference under adversarial
 * full-scale alternating input, including how often the clamp actually
 * triggers.
 *
 * The Q28 accumulator is then rescaled to Q14 by dividing by 2^14 with
 * round-half-away-from-zero (matching the §7 rounding convention used for
 * coefficient quantization), and the Q14 result is saturated to
 * [INT16_MIN, INT16_MAX] before being written into the state and returned.
 */
int16_t biquad_q14_process(biquad_q14_state_t *state, int16_t x);

#ifdef __cplusplus
}
#endif
