/* estimators_cum.h - cumulant estimator kernels (HOSA semantics). */
#ifndef SPECTRUM_ESTIMATORS_CUM_H
#define SPECTRUM_ESTIMATORS_CUM_H

#include <stddef.h>
#include <complex.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Theoretical cumulants of an ARMA model (HOSA CUMTRUE.M). ma/ar are
 * real coefficient vectors INCLUDING the leading 1 (q1/p1 = lengths).
 * nlags must already be resolved (>= 0). out: norder==2 -> (2*nlags+1)
 * doubles; else (2*nlags+1)^2 doubles, origin at the center. */
sp_status sp_cumtrue(const double *ma, long q1, const double *ar, long p1,
                     int norder, long nlags, int k, double *out);

typedef double complex sp_c128;

/* All inputs are flat, Fortran-order (column-major) complex128 buffers as
 * produced by ravel(order="F") in the Python reference. Outputs are
 * (2*maxlag+1) complex values, zero lag at index maxlag (1 value if
 * maxlag == 0 for cum2est). biased: 1 = biased, 0 = unbiased. */

sp_status sp_cum2est(const sp_c128 *y, size_t N, int maxlag, size_t nsamp,
                     int overlap, int biased, sp_c128 *out);

sp_status sp_cum2x(const sp_c128 *x, const sp_c128 *y, size_t N, size_t cols,
                   int maxlag, size_t nsamp, int overlap, int biased,
                   sp_c128 *out);

sp_status sp_cum3est(const sp_c128 *y, size_t N, int maxlag, size_t nsamp,
                     int overlap, int biased, int k1, sp_c128 *out);

sp_status sp_cum3x(const sp_c128 *x, const sp_c128 *y, const sp_c128 *z,
                   size_t N, size_t cols, int maxlag, size_t nsamp,
                   int overlap, int biased, int k1, sp_c128 *out);

sp_status sp_cum4est(const sp_c128 *y, size_t N, int maxlag, size_t nsamp,
                     int overlap, int biased, int k1, int k2, sp_c128 *out);

sp_status sp_cum4x(const sp_c128 *w, const sp_c128 *x, const sp_c128 *y,
                   const sp_c128 *z, size_t N, size_t cols, int maxlag,
                   size_t nsamp, int overlap, int biased, int k1, int k2,
                   sp_c128 *out);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRUM_ESTIMATORS_CUM_H */
