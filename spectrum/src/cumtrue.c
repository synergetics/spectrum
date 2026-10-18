/* cumtrue.c - theoretical ARMA cumulants (port of HOSA CUMTRUE.M).
 *
 * c2(i)     = sum_n h(n) h(n+i)
 * c3(i,j)   = sum_n h(n) h(n+i) h(n+j)
 * c4(i,j,k) = sum_n h(n) h(n+i) h(n+j) h(n+k)  (slice with fixed third lag)
 *
 * The impulse response uses MATLAB filter(ma, ar, [1; zeros(nsamp,1)])
 * sizing: nsamp = max(2*nlags, round(log(0.001)/log(max|pole|))).
 */
#include "common.h"
#include <complex.h>
#include <stdio.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Durand-Kerner root finding for a real monic polynomial
 * a[0] z^p + a[1] z^(p-1) + ... + a[p] = 0. Returns the maximum root
 * magnitude, or -1.0 if the iteration failed to converge. */
static double poly_max_root(const double *a, long p) {
    double complex *z = (double complex *)malloc((size_t)p * sizeof(double complex));
    double complex *az = (double complex *)malloc((size_t)p * sizeof(double complex));
    if (!z || !az) {
        free(z);
        free(az);
        return -1.0;
    }
    double complex seed = 0.4 + 0.9 * I;
    double complex cur = 1.0;
    for (long j = 0; j < p; j++) {
        z[j] = cur;
        cur *= seed;
    }
    int converged = 0;
    for (int iter = 0; iter < 500 && !converged; iter++) {
        converged = 1;
        for (long j = 0; j < p; j++) {
            /* Horner: a[0] z^p + a[1] z^(p-1) + ... + a[p] */
            double complex num = a[0];
            for (long i = 1; i <= p; i++) num = num * z[j] + a[i];
            double complex den = 1.0;
            for (long m = 0; m < p; m++) {
                if (m != j) den *= (z[j] - z[m]);
            }
            if (cabs(den) < 1e-300) {
                free(z);
                free(az);
                return -1.0;
            }
            az[j] = z[j] - num / den;
            if (cabs(az[j] - z[j]) > 1e-12) converged = 0;
        }
        memcpy(z, az, (size_t)p * sizeof(double complex));
    }
    double mx = 0.0;
    if (!converged) {
        /* accept anyway if roots look sane; else fail */
        for (long j = 0; j < p; j++) {
            if (!isfinite(creal(z[j])) || !isfinite(cimag(z[j]))) {
                free(z);
                free(az);
                return -1.0;
            }
        }
    }
    for (long j = 0; j < p; j++) {
        double r = cabs(z[j]);
        if (r > mx) mx = r;
    }
    free(z);
    free(az);
    return mx;
}

sp_status sp_cumtrue(const double *ma, long q1, const double *ar, long p1,
                     int norder, long nlags, int k, double *out) {
    long p = p1 - 1; /* AR order */
    long maxlag = nlags;
    long nlag1 = maxlag + 1;
    double *h = NULL, *z2 = NULL, *z = NULL;
    long nsamp, i, n, kk;

    if (!ma || !ar || !out) return SP_ERR_ARG;
    if (norder < 2 || norder > 4) return SP_ERR_ARG;
    if (maxlag < 0) return SP_ERR_ARG;

    /* impulse response */
    if (p == 0) {
        nsamp = q1; /* h = ma (length q1) */
        h = (double *)malloc((size_t)(nsamp > 0 ? nsamp : 1) * sizeof(double));
        if (!h) return SP_ERR_MEM;
        for (i = 0; i < nsamp; i++) h[i] = ma[i];
    } else {
        double rho = poly_max_root(ar, p);
        if (rho < 0.0) return SP_ERR_INTERNAL;
        if (rho >= 1.0) return SP_ERR_ARG; /* unstable */
        long n_rec = lround(log(0.001) / log(rho));
        nsamp = 2 * maxlag > n_rec ? 2 * maxlag : n_rec;
        h = (double *)calloc((size_t)nsamp + 1, sizeof(double));
        if (!h) return SP_ERR_MEM;
        for (n = 0; n <= nsamp; n++) {
            double acc = (n < q1) ? ma[n] : 0.0;
            for (i = 1; i <= p; i++) {
                if (n - i >= 0) acc -= ar[i] * h[n - i];
            }
            h[n] = acc / ar[0];
        }
        nsamp += 1; /* h has nsamp+1 entries */
    }

    z2 = (double *)malloc((size_t)nsamp * sizeof(double));
    z = (double *)malloc((size_t)nsamp * sizeof(double));
    if (!z2 || !z) {
        free(h);
        free(z2);
        free(z);
        return SP_ERR_MEM;
    }

    if (norder == 2) {
        double acc = 0.0;
        for (i = 0; i < nsamp; i++) acc += h[i] * h[i];
        out[nlag1 - 1] = acc;
        for (n = 1; n <= maxlag; n++) {
            double s = 0.0;
            for (i = 0; i + n < nsamp; i++) s += h[i] * h[i + n];
            out[nlag1 - 1 + n] = s;
            out[nlag1 - 1 - n] = s;
        }
        free(h);
        free(z2);
        free(z);
        return SP_OK;
    }

    /* z2(n) = h(n) h(n+k) for 4th order; z2 = h for 3rd order */
    memset(z2, 0, (size_t)nsamp * sizeof(double));
    if (norder == 3) {
        for (i = 0; i < nsamp; i++) z2[i] = h[i];
    } else if (k >= 0) {
        for (i = 0; i + k < nsamp; i++) z2[i] = h[i] * h[i + k];
    } else {
        for (i = -k; i < nsamp; i++) z2[i] = h[i] * h[i + k];
    }

    for (n = -maxlag; n <= maxlag; n++) {
        memset(z, 0, (size_t)nsamp * sizeof(double));
        if (n >= 0) {
            for (i = 0; i + n < nsamp; i++) z[i] = z2[i] * h[i + n];
        } else {
            for (i = -n; i < nsamp; i++) z[i] = z2[i] * h[i + n];
        }
        double acc = 0.0;
        for (i = 0; i < nsamp; i++) acc += z[i] * h[i];
        out[(n + nlag1 - 1) * (2 * maxlag + 1) + (nlag1 - 1)] = acc;
        for (kk = 1; kk <= maxlag; kk++) {
            double s1 = 0.0, s2 = 0.0;
            for (i = 0; i + kk < nsamp; i++) {
                s1 += z[i] * h[i + kk];
                s2 += h[i] * z[i + kk];
            }
            out[(n + nlag1 - 1) * (2 * maxlag + 1) + (nlag1 - 1 + kk)] = s1;
            out[(n + nlag1 - 1) * (2 * maxlag + 1) + (nlag1 - 1 - kk)] = s2;
        }
    }
    free(h);
    free(z2);
    free(z);
    return SP_OK;
}
