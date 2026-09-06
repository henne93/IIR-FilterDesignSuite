#include "biquad_q14.h"

void biquad_q14_init(biquad_q14_state_t *state, const q14_coeffs_t *coeffs) {
    state->coeffs = *coeffs;
    state->x1 = 0;
    state->x2 = 0;
    state->y1 = 0;
    state->y2 = 0;
}

static int16_t saturate_i16(int32_t v) {
    if (v > INT16_MAX) return INT16_MAX;
    if (v < INT16_MIN) return INT16_MIN;
    return (int16_t)v;
}

/* Overflow-checked, saturating 32-bit add/subtract -- see biquad_q14.h's
 * biquad_q14_process doc comment for why this replaces a plain int32_t sum.
 * GCC/Clang builtins only, per CONTRACTS.md §9's compiler restriction. */
static int32_t sat_add_i32(int32_t a, int32_t b) {
    int32_t result;
    if (__builtin_add_overflow(a, b, &result)) {
        return (b > 0) ? INT32_MAX : INT32_MIN;
    }
    return result;
}

static int32_t sat_sub_i32(int32_t a, int32_t b) {
    int32_t result;
    if (__builtin_sub_overflow(a, b, &result)) {
        return (b < 0) ? INT32_MAX : INT32_MIN;
    }
    return result;
}

int16_t biquad_q14_process(biquad_q14_state_t *state, int16_t x) {
    const q14_coeffs_t *c = &state->coeffs;

    int32_t p_b0 = (int32_t)c->b0 * (int32_t)x;
    int32_t p_b1 = (int32_t)c->b1 * (int32_t)state->x1;
    int32_t p_b2 = (int32_t)c->b2 * (int32_t)state->x2;
    int32_t p_a1 = (int32_t)c->a1 * (int32_t)state->y1;
    int32_t p_a2 = (int32_t)c->a2 * (int32_t)state->y2;

    int32_t acc = 0;
    acc = sat_add_i32(acc, p_b0);
    acc = sat_add_i32(acc, p_b1);
    acc = sat_add_i32(acc, p_b2);
    acc = sat_sub_i32(acc, p_a1);
    acc = sat_sub_i32(acc, p_a2);

    /* Rescale Q28 -> Q14 with round-half-away-from-zero, never right-shifting
     * a negative value directly (implementation-defined pre-C23; this codebase
     * avoids relying on it, matching the original int64_t version's style). */
    const int32_t half = (int32_t)1 << 13;  /* Q14_SCALE / 2 */
    int32_t shifted;
    if (acc >= 0) {
        shifted = sat_add_i32(acc, half) >> 14;
    } else {
        /* Negating INT32_MIN is itself UB; substitute INT32_MAX (off by one
         * part in ~2 billion) -- harmless since this path is already
         * saturated/clipped and the final int16 clamp dominates anyway. */
        int32_t neg = (acc == INT32_MIN) ? INT32_MAX : -acc;
        shifted = -(sat_add_i32(neg, half) >> 14);
    }

    int16_t y = saturate_i16(shifted);

    state->x2 = state->x1;
    state->x1 = x;
    state->y2 = state->y1;
    state->y1 = y;

    return y;
}
