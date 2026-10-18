/* wigner.c - Wigner-Ville higher-order distributions.
 *
 * Definitions follow spectrum/wig.py (HOSA manual + Nikias &
 * Petropulu conventions); the pure-Python module is the reference.
 */
#include "wigner.h"
#include "fft/fft_iface.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

static sp_status wig2_kernel(const sp_c128 *x, size_t N, size_t nfft,
                              size_t inc, int complex_version,
                              sp_c128 *W /* n_time * nfft */) {
    size_t n_time = (N + inc - 1) / inc;
    size_t M = nfft / 2;
    sp_c128 *kern = (sp_c128 *)calloc(nfft, sizeof(sp_c128));
    if (!kern) return SP_ERR_MEM;
    for (size_t it = 0; it < n_time; it++) {
        size_t n0 = it * inc;
        memset(kern, 0, nfft * sizeof(sp_c128));
        for (long m = -(long)M; m < (long)M; m++) {
            size_t idx = (m >= 0) ? (size_t)m : nfft + (size_t)m;
            long ai = (long)n0 + m;
            long bi = (long)n0 - m;
            sp_c128 a = (ai >= 0 && (size_t)ai < N) ? x[ai] : 0.0;
            sp_c128 b = (bi >= 0 && (size_t)bi < N) ? x[bi] : 0.0;
            kern[idx] = complex_version ? a * conj(b) : a * b;
        }
        sp_c128 *row = W + it * nfft;
        memcpy(row, kern, nfft * sizeof(sp_c128));
        if (sp_fft_forward(row, nfft) != 0) {
            free(kern);
            return SP_ERR_INTERNAL;
        }
    }
    free(kern);
    /* real version: keep only the real part */
    if (!complex_version) {
        for (size_t i = 0; i < n_time * nfft; i++) W[i] = creal(W[i]);
    }
    return SP_OK;
}

sp_status sp_wig2(const sp_c128 *x, size_t N, size_t nfft, size_t inc,
                  int complex_version, sp_c128 *W) {
    if (!x || !W || nfft == 0 || inc == 0) return SP_ERR_ARG;
    return wig2_kernel(x, N, nfft, inc, complex_version, W);
}

sp_status sp_wig4(const sp_c128 *x, size_t N, size_t nfft, size_t inc,
                  int complex_version, sp_c128 *W /* n_time*nfft*nfft */) {
    size_t n_time = (N + inc - 1) / inc;
    size_t M = nfft / 2;
    sp_c128 *kern = (sp_c128 *)calloc(nfft * nfft, sizeof(sp_c128));
    sp_c128 *col = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    if (!kern || !col) {
        free(kern);
        free(col);
        return SP_ERR_MEM;
    }
    for (size_t it = 0; it < n_time; it++) {
        size_t n0 = it * inc;
        memset(kern, 0, nfft * nfft * sizeof(sp_c128));
        for (long k1 = -(long)M; k1 < (long)M; k1++) {
            size_t i1 = (k1 >= 0) ? (size_t)k1 : nfft + (size_t)k1;
            for (long k2 = -(long)M; k2 < (long)M; k2++) {
                size_t i2 = (k2 >= 0) ? (size_t)k2 : nfft + (size_t)k2;
                long ai = (long)n0 + k1;
                long bi = (long)n0 + k2;
                sp_c128 a = (ai >= 0 && (size_t)ai < N) ? x[ai] : 0.0;
                sp_c128 b = (bi >= 0 && (size_t)bi < N) ? x[bi] : 0.0;
                long n2 = (long)n0 + k1 + k2;
                sp_c128 c = (n2 >= 0 && (size_t)n2 < N) ? x[n2] : 0.0;
                kern[i1 * nfft + i2] =
                    complex_version ? a * b * conj(c) : a * b * c;
            }
        }
        /* fft2: rows then columns */
        sp_c128 *slice = W + it * nfft * nfft;
        for (size_t r = 0; r < nfft; r++) {
            memcpy(slice + r * nfft, kern + r * nfft, nfft * sizeof(sp_c128));
            if (sp_fft_forward(slice + r * nfft, nfft) != 0) {
                free(kern);
                free(col);
                return SP_ERR_INTERNAL;
            }
        }
        for (size_t c = 0; c < nfft; c++) {
            for (size_t r = 0; r < nfft; r++) col[r] = slice[r * nfft + c];
            if (sp_fft_forward(col, nfft) != 0) {
                free(kern);
                free(col);
                return SP_ERR_INTERNAL;
            }
            for (size_t r = 0; r < nfft; r++) slice[r * nfft + c] = col[r];
        }
    }
    free(kern);
    free(col);
    return SP_OK;
}
