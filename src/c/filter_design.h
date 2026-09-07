#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q14_SCALE 16384

/* Storage width is int16_t: a full-domain sweep of every LP/HP/BP/AP design in
 * the supported parameter range (CONTRACTS.md §5) gives a true worst-case
 * |coefficient| of 1.9958 (AP a1/b1), which fits int16 Q14 (max representable
 * ~1.99994) with ~0.004 headroom -- tight but real, and locked by an
 * automated regression sweep (tests/test_native_coefficients.py). */
typedef struct {
    int16_t b0, b1, b2, a1, a2;   /* Q14 fixed-point, scale = 16384 */
} q14_coeffs_t;

/* Return 0 on success, negative on invalid input (defense-in-depth only —
   Python validates first per CONTRACTS.md §5 and should never trigger these
   in practice).
   Error codes: 0 success, -1 frequency out of (0, fs/2), -2 fs <= 0,
   -3 f_low >= f_high (BP only), -4 q out of (0, inf) (AP, PK). */
int filter_design_lp(float fc, float fs, q14_coeffs_t *out);
int filter_design_hp(float fc, float fs, q14_coeffs_t *out);
int filter_design_bp(float f_low, float f_high, float fs, q14_coeffs_t *out);
int filter_design_ap(float fc, float fs, float q, q14_coeffs_t *out);
int filter_design_pk(float fc, float fs, float q, float gain_db, q14_coeffs_t *out);

#ifdef __cplusplus
}
#endif
