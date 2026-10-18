/* estimators_bispec.h - bispectrum-family kernels (HOSA semantics). */
#ifndef SPECTRUM_ESTIMATORS_BISPEC_H
#define SPECTRUM_ESTIMATORS_BISPEC_H

#include <stddef.h>
#include "estimators_cum.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shared segmentation meta: resolves defaults exactly like the corrected
 * Python estimators. Inputs: ly (rows), cols, user nsamp (<=0 = default),
 * user overlap (percent), user nfft. Outputs: resolved nsamp, resolved nfft
 * (bumped up to the next power of two >= nsamp when too small). Returns
 * SP_ERR_ARG on degenerate input. */
sp_status sp_bispec_meta(size_t ly, size_t cols, long nsamp_in, int overlap_in,
                         long nfft_in, size_t *nsamp_out, size_t *nfft_out);

/* Frequency-domain smoothing window specification for bispectrumd/dx. */
typedef struct {
    int kind; /* 0 = Rao-Gabr optimal window of size `opt_size` (default 5)
                 1 = 1-D vector (zero-padded hankel construction)
                 2 = 2-D matrix (used directly when square + odd) */
    int opt_size;
    const double *vec;
    long vec_len;
    const double *mat; /* row-major, mat_dim x mat_dim */
    long mat_dim;
} sp_wind_spec;

/* Bicoherence (direct method). bic_out: nfft*nfft (row-major) complex,
 * waxis_out: nfft doubles. wind_t: time-domain window of length nsamp, or
 * NULL for the default hanning(nsamp). */
sp_status sp_bicoherx(const sp_c128 *w, const sp_c128 *x, const sp_c128 *y,
                      size_t N, size_t cols, size_t nsamp, size_t nfft,
                      const double *wind_t, int overlap, int symmetric,
                      sp_c128 *bic_out, double *waxis_out);

/* Bispectrum, direct method (cross version; pass x=y=z for auto). */
sp_status sp_bispecdx(const sp_c128 *x, const sp_c128 *y, const sp_c128 *z,
                      size_t N, size_t cols, size_t nsamp, size_t nfft,
                      const sp_wind_spec *wind, int overlap, int symmetric,
                      sp_c128 *bspec_out, double *waxis_out);

/* Bispectrum, indirect method (FFT of windowed 3rd-order cumulants).
 * wind_mode: 0 = Parzen lag window, 1 = hexagonal unity, -1 = no window. */
sp_status sp_bispeci(const sp_c128 *y, size_t N, long nlag, size_t nsamp,
                     int overlap, int biased, size_t nfft, int wind_mode,
                     sp_c128 *bspec_out, double *waxis_out);

/* Trispectrum (reduced-grid direct method) and tricoherence.
 * tspec_out: mfi^3 complex, waxis_out: mfi doubles where
 * mfi = min(nfft/2 + 1, nfft/8). window_type: 0 rect, 1 hanning,
 * 2 hamming, 3 blackman. */
sp_status sp_trispectrum(const sp_c128 *y, size_t N, size_t nsamp, size_t nfft,
                         int window_type, int overlap, sp_c128 *tspec_out,
                         double *waxis_out, size_t *mfi_out);

sp_status sp_tricoherence(const sp_c128 *y, size_t N, size_t nsamp,
                          size_t nfft, int window_type, int overlap,
                          double *tricoh_out, double *waxis_out,
                          size_t *mfi_out);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRUM_ESTIMATORS_BISPEC_H */
