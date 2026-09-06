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
 */
typedef struct {
    q14_coeffs_t coeffs;
    int32_t x1, x2;   /* x[n-1], x[n-2] */
    int32_t y1, y2;   /* y[n-1], y[n-2] */
} biquad_q14_state_t;

void biquad_q14_init(biquad_q14_state_t *state, const q14_coeffs_t *coeffs);

/*
 * Processes one Q14 input sample through
 *   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 * (Direct Form 1; minus signs on the a-terms match the CONTRACTS.md §2
 * denominator convention 1 + a1*z^-1 + a2*z^-2) and returns y[n].
 *
 * Each of the 5 coefficient*sample products is a Q28 value; all 5 are
 * summed in a widened int64_t accumulator so no partial sum can overflow.
 * The Q28 accumulator is then rescaled to Q14 by dividing by 2^14 with
 * round-half-away-from-zero (matching the §7 rounding convention used for
 * coefficient quantization), and the Q14 result is saturated to
 * [INT32_MIN, INT32_MAX] before being written into the state and returned.
 * Saturation is a defensive guard: for coefficients produced by the
 * filter_design_* functions from any input inside the CONTRACTS.md §5
 * parameter domain, driven by a bounded input sample, it is not expected
 * to trigger.
 */
int32_t biquad_q14_process(biquad_q14_state_t *state, int32_t x);

#ifdef __cplusplus
}
#endif
