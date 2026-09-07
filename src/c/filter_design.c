/* Q14 coefficient design for LP/HP/BP/AP — equations per CONTRACTS.md §3.
 *
 * All arithmetic here is C `float` (32-bit), matching CONCEPT.md's
 * tanf/float usage and CONTRACTS.md §4's "C Q14 coefficients = the same §3
 * equations evaluated in C float, then Q14-quantized" rule. All functions
 * are pure/stateless per CONTRACTS.md §8.
 */

#include "filter_design.h"

#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

/* Round-half-away-from-zero (roundf) into Q14, per CONTRACTS.md §7.
 * Storage is int16_t. Unlike the original int32_t storage (which had huge
 * headroom), int16_t is genuinely tight: a full-domain sweep of every
 * LP/HP/BP/AP design gives a true worst-case |coefficient| of 1.9958 (AP
 * a1/b1), i.e. ~32712 in Q14 units, ~55 counts (~0.17%) under INT16_MAX
 * (32767) -- never actually reached in-domain (regression-guarded by
 * tests/test_native_coefficients.py), but the saturation below is no longer
 * purely hypothetical the way it was at int32_t width. */
static int16_t q14_quantize(float x) {
    double scaled = (double)roundf(x * (float)Q14_SCALE);
    if (scaled >= (double)INT16_MAX) {
        return INT16_MAX;
    }
    if (scaled <= (double)INT16_MIN) {
        return INT16_MIN;
    }
    return (int16_t)scaled;
}

static int check_fs(float fs) {
    return (fs > 0.0f) ? 0 : -2;
}

static int check_edge(float f, float fs) {
    return (f > 0.0f && f < fs * 0.5f) ? 0 : -1;
}

static void store(q14_coeffs_t *out, float b0, float b1, float b2, float a1, float a2) {
    out->b0 = q14_quantize(b0);
    out->b1 = q14_quantize(b1);
    out->b2 = q14_quantize(b2);
    out->a1 = q14_quantize(a1);
    out->a2 = q14_quantize(a2);
}

int filter_design_lp(float fc, float fs, q14_coeffs_t *out) {
    int rc;
    if ((rc = check_fs(fs)) != 0) return rc;
    if ((rc = check_edge(fc, fs)) != 0) return rc;

    float K = tanf((float)M_PI * fc / fs);
    float norm = 1.0f + (float)M_SQRT2 * K + K * K;
    float b0 = (K * K) / norm;
    float b1 = 2.0f * b0;
    float b2 = b0;
    float a1 = 2.0f * (K * K - 1.0f) / norm;
    float a2 = (K * K - (float)M_SQRT2 * K + 1.0f) / norm;

    store(out, b0, b1, b2, a1, a2);
    return 0;
}

int filter_design_hp(float fc, float fs, q14_coeffs_t *out) {
    int rc;
    if ((rc = check_fs(fs)) != 0) return rc;
    if ((rc = check_edge(fc, fs)) != 0) return rc;

    float K = tanf((float)M_PI * fc / fs);
    float norm = 1.0f + (float)M_SQRT2 * K + K * K;
    float b0 = 1.0f / norm;
    float b1 = -2.0f * b0;
    float b2 = b0;
    float a1 = 2.0f * (K * K - 1.0f) / norm;
    float a2 = (K * K - (float)M_SQRT2 * K + 1.0f) / norm;

    store(out, b0, b1, b2, a1, a2);
    return 0;
}

int filter_design_bp(float f_low, float f_high, float fs, q14_coeffs_t *out) {
    int rc;
    if ((rc = check_fs(fs)) != 0) return rc;
    if ((rc = check_edge(f_low, fs)) != 0) return rc;
    if ((rc = check_edge(f_high, fs)) != 0) return rc;
    if (!(f_low < f_high)) return -3;

    float K_low = tanf((float)M_PI * f_low / fs);
    float K_high = tanf((float)M_PI * f_high / fs);
    float wc2 = K_low * K_high;
    float BW = K_high - K_low;
    float norm = 1.0f + BW + wc2;
    float b0 = BW / norm;
    float b1 = 0.0f;
    float b2 = -BW / norm;
    float a1 = 2.0f * (wc2 - 1.0f) / norm;
    float a2 = (1.0f - BW + wc2) / norm;

    store(out, b0, b1, b2, a1, a2);
    return 0;
}

int filter_design_ap(float fc, float fs, float q, q14_coeffs_t *out) {
    int rc;
    if ((rc = check_fs(fs)) != 0) return rc;
    if ((rc = check_edge(fc, fs)) != 0) return rc;
    if (!(q > 0.0f)) return -4;

    float w0 = 2.0f * (float)M_PI * fc / fs;
    float alpha = sinf(w0) / (2.0f * q);
    float norm = 1.0f + alpha;
    float a1 = -2.0f * cosf(w0) / norm;
    float a2 = (1.0f - alpha) / norm;
    float b0 = a2;
    float b1 = a1;
    float b2 = 1.0f;

    store(out, b0, b1, b2, a1, a2);
    return 0;
}

/* Peaking EQ (bell boost/cut), CONTRACTS.md §3: bilinear transform of the
 * analog prototype H(s) = (s^2 + (A/Q)s + 1) / (s^2 + s/(A*Q) + 1),
 * A = 10^(gain_db/40), via s = (1/K)*(1-z^-1)/(1+z^-1), K = tan(pi*fc/fs) --
 * same prewarping family as LP/HP/BP, unlike AP's un-prewarped w0. b1 and a1
 * are identical because the z^-1 term (2*(K^2-1)) is common to numerator and
 * denominator before normalization -- not a bug. */
int filter_design_pk(float fc, float fs, float q, float gain_db, q14_coeffs_t *out) {
    int rc;
    if ((rc = check_fs(fs)) != 0) return rc;
    if ((rc = check_edge(fc, fs)) != 0) return rc;
    if (!(q > 0.0f)) return -4;

    float K = tanf((float)M_PI * fc / fs);
    float A = powf(10.0f, gain_db / 40.0f);
    float K2 = K * K;
    float alpha_num = (A / q) * K;
    float alpha_den = K / (A * q);
    float norm = K2 + alpha_den + 1.0f;
    float b0 = (K2 + alpha_num + 1.0f) / norm;
    float b1 = 2.0f * (K2 - 1.0f) / norm;
    float b2 = (K2 - alpha_num + 1.0f) / norm;
    float a1 = b1;
    float a2 = (K2 - alpha_den + 1.0f) / norm;

    store(out, b0, b1, b2, a1, a2);
    return 0;
}
