#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q14_SCALE 16384

typedef struct {
    int32_t b0, b1, b2, a1, a2;   /* Q14 fixed-point, scale = 16384 */
} q14_coeffs_t;

/* Return 0 on success, negative on invalid input (defense-in-depth only —
   Python validates first per CONTRACTS.md §5 and should never trigger these
   in practice).
   Error codes: 0 success, -1 frequency out of (0, fs/2), -2 fs <= 0,
   -3 f_low >= f_high (BP only), -4 q out of (0, inf) (AP only). */
int filter_design_lp(float fc, float fs, q14_coeffs_t *out);
int filter_design_hp(float fc, float fs, q14_coeffs_t *out);
int filter_design_bp(float f_low, float f_high, float fs, q14_coeffs_t *out);
int filter_design_ap(float fc, float fs, float q, q14_coeffs_t *out);

#ifdef __cplusplus
}
#endif
