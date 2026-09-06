#include "biquad_q14.h"

void biquad_q14_init(biquad_q14_state_t *state, const q14_coeffs_t *coeffs) {
    state->coeffs = *coeffs;
    state->x1 = 0;
    state->x2 = 0;
    state->y1 = 0;
    state->y2 = 0;
}

static int32_t saturate_i32(int64_t v) {
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return (int32_t)v;
}

int32_t biquad_q14_process(biquad_q14_state_t *state, int32_t x) {
    const q14_coeffs_t *c = &state->coeffs;

    int64_t acc = (int64_t)c->b0 * (int64_t)x
                + (int64_t)c->b1 * (int64_t)state->x1
                + (int64_t)c->b2 * (int64_t)state->x2
                - (int64_t)c->a1 * (int64_t)state->y1
                - (int64_t)c->a2 * (int64_t)state->y2;

    const int64_t half = (int64_t)1 << 13;  /* Q14_SCALE / 2 */
    int64_t y_wide = (acc >= 0) ? ((acc + half) >> 14) : -(((-acc) + half) >> 14);

    int32_t y = saturate_i32(y_wide);

    state->x2 = state->x1;
    state->x1 = x;
    state->y2 = state->y1;
    state->y1 = y;

    return y;
}
