/* estimators_bispec.c - bispectrum-family kernels, exact ports of the
 * corrected Python estimators (bicoherence.py, bicoherencex.py,
 * bispectrumd.py, bispectrumdx.py, bispectrumi.py, trispectrum.py).
 *
 * Inputs are Fortran-order flat complex128 buffers (== ravel(order="F")).
 * All 2-D/3-D outputs are row-major (numpy C order). Multi-column inputs
 * are segmented along the flattened column-major stream exactly like the
 * Python reference (ravel(order="F") then contiguous nsamp slices).
 * FFTs go through fft_iface (numpy conventions).
 */
#include "estimators_bispec.h"
#include "fft/fft_iface.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static sp_c128 vec_mean(const sp_c128 *v, size_t n) {
    double sr = 0.0, si = 0.0;
    for (size_t i = 0; i < n; i++) {
        sr += creal(v[i]);
        si += cimag(v[i]);
    }
    return (sr / (double)n) + (si / (double)n) * I;
}

static void build_hann(double *w, size_t M) {
    /* numpy.hanning: 0.5 - 0.5*cos(2*pi*k/(M-1)); ones(1) for M == 1 */
    if (M == 0) return;
    if (M == 1) {
        w[0] = 1.0;
        return;
    }
    for (size_t k = 0; k < M; k++)
        w[k] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)k / (double)(M - 1));
}

static void build_hamming(double *w, size_t M) {
    if (M == 0) return;
    if (M == 1) {
        w[0] = 1.0;
        return;
    }
    for (size_t k = 0; k < M; k++)
        w[k] = 0.54 - 0.46 * cos(2.0 * M_PI * (double)k / (double)(M - 1));
}

static void build_blackman(double *w, size_t M) {
    if (M == 0) return;
    if (M == 1) {
        w[0] = 1.0;
        return;
    }
    for (size_t k = 0; k < M; k++) {
        double a = 2.0 * M_PI * (double)k / (double)(M - 1);
        w[k] = 0.42 - 0.5 * cos(a) + 0.08 * cos(2.0 * a);
    }
}

static void waxis_fill(double *waxis, size_t nfft) {
    if (nfft % 2 == 0) {
        for (size_t i = 0; i < nfft; i++)
            waxis[i] = ((double)(long)i - (double)nfft / 2.0) / (double)nfft;
    } else {
        for (size_t i = 0; i < nfft; i++)
            waxis[i] = ((double)(long)i - (double)(nfft - 1) / 2.0) /
                       (double)nfft;
    }
}

static void fftshift2d(sp_c128 *a, size_t n) {
    /* np.fft.fftshift on an n x n array: roll both axes by n/2 */
    size_t half = n / 2;
    sp_c128 *tmp = (sp_c128 *)malloc(n * n * sizeof(sp_c128));
    if (!tmp) return;
    for (size_t i = 0; i < n; i++) {
        size_t si = (i + n - half) % n;
        for (size_t j = 0; j < n; j++) {
            size_t sj = (j + n - half) % n;
            tmp[i * n + j] = a[si * n + sj];
        }
    }
    memcpy(a, tmp, n * n * sizeof(sp_c128));
    free(tmp);
}

sp_status sp_bispec_meta(size_t ly, size_t cols, long nsamp_in, int overlap_in,
                         long nfft_in, size_t *nsamp_out, size_t *nfft_out) {
    size_t nrecs = cols ? cols : 1;
    int overlap = overlap_in;
    long nsamp = nsamp_in;
    long nfft = nfft_in;

    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    if (nrecs > 1) overlap = 0;
    if (nrecs > 1) nsamp = (long)ly;
    if (nrecs == 1 && nsamp <= 0)
        nsamp = sp_fix((double)ly / (8.0 - 7.0 * overlap / 100.0));
    if (nsamp <= 0) return SP_ERR_ARG;

    if (nfft < nsamp) nfft = (long)pow(2.0, ceil(log2((double)nsamp)));
    if (nfft <= 0) return SP_ERR_ARG;

    *nsamp_out = (size_t)nsamp;
    *nfft_out = (size_t)nfft;
    return SP_OK;
}

static sp_status seg_counts(size_t ly, size_t cols, size_t nsamp, int overlap,
                            size_t *ov_out, size_t *nadv_out,
                            size_t *nrecs_out) {
    size_t nrecs = cols ? cols : 1;
    size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
    size_t nadv = nsamp - ov;
    if (nadv == 0) return SP_ERR_ARG;
    nrecs = (size_t)sp_fix(((double)ly * (double)nrecs - (double)ov) /
                           (double)nadv);
    if (nrecs == 0) return SP_ERR_ARG;
    *ov_out = ov;
    *nadv_out = nadv;
    *nrecs_out = nrecs;
    return SP_OK;
}

/* 2-D "same" convolution with a square kernel (scipy convolve2d). */
static int conv2d_same(sp_c128 *out, const sp_c128 *A, const double *K,
                       size_t n, size_t w) {
    size_t half = w / 2;
    /* scipy.signal.convolve2d(mode='same'): out[i,j] =
     * sum_{s,t} A[i+half-s, j+half-t] * K[s,t] (convolution flips K) */
    SP_OMP_PARALLEL_FOR
    for (long long ii = 0; ii < (long long)n; ii++) {
        size_t i = (size_t)ii;
        for (size_t j = 0; j < n; j++) {
            sp_c128 acc = 0.0;
            for (size_t s = 0; s < w; s++) {
                long long ai = (long long)i + (long long)half - (long long)s;
                if (ai < 0 || (size_t)ai >= n) continue;
                for (size_t t = 0; t < w; t++) {
                    long long aj = (long long)j + (long long)half - (long long)t;
                    if (aj < 0 || (size_t)aj >= n) continue;
                    acc += A[(size_t)ai * n + (size_t)aj] * K[s * w + t];
                }
            }
            out[i * n + j] = acc;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* bicoherence (direct method)                                         */
/* ------------------------------------------------------------------ */

sp_status sp_bicoherx(const sp_c128 *w, const sp_c128 *x, const sp_c128 *y,
                      size_t N, size_t cols, size_t nsamp, size_t nfft,
                      const double *wind_t, int overlap, int symmetric,
                      sp_c128 *bic_out, double *waxis_out) {
    size_t ov, nadv, nrecs;
    double *hannbuf = NULL;
    const double *wind = wind_t;
    sp_c128 *Pww = NULL, *Pxx = NULL, *Pyy = NULL;
    sp_c128 *Wf = NULL, *Xf = NULL, *Yf = NULL;
    sp_c128 *seg = NULL;
    sp_status st;

    if (!w || !x || !y || !bic_out || !waxis_out) return SP_ERR_ARG;
    memset(bic_out, 0, nfft * nfft * sizeof(sp_c128)); /* buffers arrive
        * uninitialized from PyArray_SimpleNew; we accumulate in place */
    st = seg_counts(N / (cols ? cols : 1), cols, nsamp, overlap, &ov, &nadv,
                    &nrecs);
    if (st != SP_OK) return st;

    if (!wind) {
        hannbuf = (double *)malloc(nsamp * sizeof(double));
        if (!hannbuf) return SP_ERR_MEM;
        build_hann(hannbuf, nsamp);
        wind = hannbuf;
    }

    Pww = (sp_c128 *)calloc(nfft, sizeof(sp_c128));
    Pxx = (sp_c128 *)calloc(nfft, sizeof(sp_c128));
    Pyy = (sp_c128 *)calloc(nfft, sizeof(sp_c128));
    Wf = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    Xf = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    Yf = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    seg = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
    if (!Pww || !Pxx || !Pyy || !Wf || !Xf || !Yf || !seg) {
        free(hannbuf);
        free(Pww);
        free(Pxx);
        free(Pyy);
        free(Wf);
        free(Xf);
        free(Yf);
        free(seg);
        return SP_ERR_MEM;
    }

    for (size_t k = 0; k < nrecs; k++) {
        size_t base = k * nadv;
        if (base + nsamp > N) {
            st = SP_ERR_SHAPE;
            goto done;
        }
        const sp_c128 *spec_in[3] = {w, x, y};
        sp_c128 *spec_out[3] = {Wf, Xf, Yf};
        for (int s = 0; s < 3; s++) {
            memcpy(seg, spec_in[s] + base, nsamp * sizeof(sp_c128));
            sp_c128 m = vec_mean(seg, nsamp);
            for (size_t n = 0; n < nsamp; n++) seg[n] = (seg[n] - m) * wind[n];
            memset(spec_out[s], 0, nfft * sizeof(sp_c128));
            memcpy(spec_out[s], seg, nsamp * sizeof(sp_c128));
            if (sp_fft_forward(spec_out[s], nfft) != 0) {
                st = SP_ERR_INTERNAL;
                goto done;
            }
            for (size_t n = 0; n < nfft; n++) spec_out[s][n] /= (double)nsamp;
        }
        for (size_t n = 0; n < nfft; n++) {
            Pww[n] += Wf[n] * conj(Wf[n]);
            Pxx[n] += Xf[n] * conj(Xf[n]);
            Pyy[n] += Yf[n] * conj(Yf[n]);
        }

        /* bic(i,j) += Wf(i) Xf(j) conj(Yf((i+j) mod nfft)); exact (i,j)
         * symmetry only in the auto case -> lower triangle + mirror */
        if (symmetric) {
            SP_OMP_PARALLEL_FOR
            for (long long ii = 0; ii < (long long)nfft; ii++) {
                size_t i = (size_t)ii;
                sp_c128 wi = Wf[i];
                size_t split = i < nfft - i ? i : nfft - i;
                for (size_t j = 0; j < split; j++) {
                    size_t ij = i + j;
                    sp_c128 v = wi * Xf[j] * conj(Yf[ij]);
                    bic_out[i * nfft + j] += v;
                    bic_out[j * nfft + i] += v;
                }
                if (split == i) { /* j == i boundary, no wrap */
                    sp_c128 v = wi * Xf[i] * conj(Yf[2 * i % nfft]);
                    bic_out[i * nfft + i] += v;
                } else {
                    for (size_t j = split; j <= i; j++) {
                        size_t ij = i + j - nfft;
                        sp_c128 v = wi * Xf[j] * conj(Yf[ij]);
                        bic_out[i * nfft + j] += v;
                        if (j != i) bic_out[j * nfft + i] += v;
                    }
                }
            }
        } else {
            /* MATLAB BICOHERX: (i,j) = Wf[i] * Xf[j] * conj(Yf[ij]);
             * the wrapped index (i+j) mod nfft wraps once per row */
            SP_OMP_PARALLEL_FOR
            for (long long ii = 0; ii < (long long)nfft; ii++) {
                size_t i = (size_t)ii;
                sp_c128 wi = Wf[i];
                sp_c128 *rowp = bic_out + i * nfft;
                size_t split = nfft - i;
                for (size_t j = 0; j < split; j++)
                    rowp[j] += wi * Xf[j] * conj(Yf[i + j]);
                for (size_t j = split; j < nfft; j++)
                    rowp[j] += wi * Xf[j] * conj(Yf[i + j - nfft]);
            }
        }
    }

    {
        /* (i,j) denominator: Pww[i] * Pxx[j] * Pyy[(i+j) mod nfft] */
        double inv = 1.0 / (double)nrecs;
        for (size_t i = 0; i < nfft; i++) {
            double pw = cabs(Pww[i]) * inv;
            for (size_t j = 0; j < nfft; j++) {
                size_t ij = (i + j) % nfft;
                double denom = pw * (cabs(Pxx[j]) * inv) * (cabs(Pyy[ij]) * inv);
                double num = cabs(bic_out[i * nfft + j] * inv);
                num = num * num;
                bic_out[i * nfft + j] = denom > 0.0 ? num / denom : 0.0;
            }
        }
    }
    fftshift2d(bic_out, nfft);
    waxis_fill(waxis_out, nfft);
    st = SP_OK;

done:
    free(hannbuf);
    free(Pww);
    free(Pxx);
    free(Pyy);
    free(Wf);
    free(Xf);
    free(Yf);
    free(seg);
    return st;
}

/* ------------------------------------------------------------------ */
/* bispectrum, direct method (cross version)                          */
/* ------------------------------------------------------------------ */

sp_status sp_bispecdx(const sp_c128 *x, const sp_c128 *y, const sp_c128 *z,
                      size_t N, size_t cols, size_t nsamp, size_t nfft,
                      const sp_wind_spec *wind, int overlap, int symmetric,
                      sp_c128 *bspec_out, double *waxis_out) {
    size_t ov, nadv, nrecs;
    size_t winsize = 1;
    double *opwind_d = NULL;
    sp_c128 *opwind_c = NULL; /* real kernel stored as complex workspace */
    sp_c128 *Xf = NULL, *Yf = NULL, *Zf = NULL, *seg = NULL, *tmp2 = NULL;
    sp_status st;

    if (!x || !y || !z || !bspec_out || !waxis_out) return SP_ERR_ARG;
    memset(bspec_out, 0, nfft * nfft * sizeof(sp_c128));
    st = seg_counts(N / (cols ? cols : 1), cols, nsamp, overlap, &ov, &nadv,
                    &nrecs);
    if (st != SP_OK) return st;

    /* ---- build the frequency-domain smoothing window ---- */
    if (!wind || wind->kind == 0) {
        int ws = wind ? wind->opt_size : 5;
        if (ws < 0) ws = 5;
        ws = ws - (ws % 2) + 1; /* odd */
        if (ws > 1) {
            size_t w2 = (size_t)ws;
            double mwind = floor((double)nfft / (double)w2);
            double lby2 = (double)(w2 - 1) / 2.0;
            opwind_d = (double *)malloc(w2 * w2 * sizeof(double));
            if (!opwind_d) return SP_ERR_MEM;
            for (size_t i = 0; i < w2; i++) {
                double ti = (double)i - lby2;
                for (size_t j = 0; j < w2; j++) {
                    double tj = (double)j - lby2;
                    double q = 2.0 * mwind / (double)nfft;
                    double v = 1.0 - q * q * (ti * ti + tj * tj + ti * tj);
                    double hex = fabs(ti) + fabs(tj) + fabs(ti + tj);
                    opwind_d[i * w2 + j] = (hex < (double)ws) ? v : 0.0;
                }
            }
            for (size_t i = 0; i < w2 * w2; i++)
                opwind_d[i] *= 4.0 * mwind * mwind / (7.0 * M_PI * M_PI);
            winsize = w2;
        } else {
            winsize = 1;
        }
    } else if (wind->kind == 1) {
        long L = wind->vec_len;
        if (L > 0) {
            long P = 2 * L - 1;
            double *padded = (double *)calloc((size_t)P, sizeof(double));
            double *windf = (double *)malloc((size_t)P * sizeof(double));
            double *hank = (double *)malloc((size_t)P * (size_t)P * sizeof(double));
            opwind_d = (double *)malloc((size_t)P * (size_t)P * sizeof(double));
            if (!padded || !windf || !hank || !opwind_d) {
                free(padded);
                free(windf);
                free(hank);
                free(opwind_d);
                return SP_ERR_MEM;
            }
            memcpy(padded, wind->vec, (size_t)L * sizeof(double));
            /* windf = [wind reversed without first, wind] */
            for (long i = 0; i < L - 1; i++) windf[i] = wind->vec[L - 1 - i];
            for (long i = 0; i < L; i++) windf[L - 1 + i] = wind->vec[i];
            /* hankel(flipud(padded), padded): c[k] = padded[P-1-k];
             * H(i,j) = c[i+j] if i+j <= P-1 else r[i+j-P+1] */
            for (long i = 0; i < P; i++)
                for (long j = 0; j < P; j++) {
                    long s = i + j;
                    hank[i * P + j] = (s <= P - 1) ? padded[P - 1 - s]
                                                   : padded[s - P + 1];
                }
            for (long i = 0; i < P; i++)
                for (long j = 0; j < P; j++)
                    opwind_d[i * P + j] = windf[i] * windf[j] * hank[i * P + j];
            winsize = (size_t)P;
            free(padded);
            free(windf);
            free(hank);
        }
    } else if (wind->kind == 2) {
        long m = wind->mat_dim;
        if (m > 0 && wind->mat) {
            /* binding guarantees square + odd for kind 2 */
            opwind_d = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
            if (!opwind_d) return SP_ERR_MEM;
            memcpy(opwind_d, wind->mat, (size_t)m * (size_t)m * sizeof(double));
            winsize = (size_t)m;
        } else {
            winsize = 1;
        }
    }

    /* the smoothing kernel is real; conv2d_same takes double* */
    Xf = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    Yf = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    Zf = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    seg = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
    tmp2 = (sp_c128 *)malloc(nfft * nfft * sizeof(sp_c128));
    if (!Xf || !Yf || !Zf || !seg || !tmp2) {
        free(opwind_d);
        free(opwind_c);
        free(Xf);
        free(Yf);
        free(Zf);
        free(seg);
        free(tmp2);
        return SP_ERR_MEM;
    }

    for (size_t k = 0; k < nrecs; k++) {
        size_t base = k * nadv;
        if (base + nsamp > N) {
            st = SP_ERR_SHAPE;
            goto done;
        }
        const sp_c128 *spec_in[3] = {x, y, z};
        sp_c128 *spec_out[3] = {Xf, Yf, Zf};
        for (int s = 0; s < 3; s++) {
            memcpy(seg, spec_in[s] + base, nsamp * sizeof(sp_c128));
            sp_c128 m = vec_mean(seg, nsamp);
            for (size_t n = 0; n < nsamp; n++) seg[n] -= m;
            memset(spec_out[s], 0, nfft * sizeof(sp_c128));
            memcpy(spec_out[s], seg, nsamp * sizeof(sp_c128));
            if (sp_fft_forward(spec_out[s], nfft) != 0) {
                st = SP_ERR_INTERNAL;
                goto done;
            }
            for (size_t n = 0; n < nfft; n++) spec_out[s][n] /= (double)nsamp;
        }
        /* Bspec(i,j) += Xf(i) Yf(j) conj(Zf((i+j) mod nfft)) */
        if (symmetric) {
            SP_OMP_PARALLEL_FOR
            for (long long ii = 0; ii < (long long)nfft; ii++) {
                size_t i = (size_t)ii;
                sp_c128 xi = Xf[i];
                size_t split = i < nfft - i ? i : nfft - i;
                for (size_t j = 0; j < split; j++) {
                    size_t ij = i + j;
                    sp_c128 v = xi * Yf[j] * conj(Zf[ij]);
                    bspec_out[i * nfft + j] += v;
                    bspec_out[j * nfft + i] += v;
                }
                if (split == i) {
                    sp_c128 v = xi * Yf[i] * conj(Zf[2 * i % nfft]);
                    bspec_out[i * nfft + i] += v;
                } else {
                    for (size_t j = split; j <= i; j++) {
                        size_t ij = i + j - nfft;
                        sp_c128 v = xi * Yf[j] * conj(Zf[ij]);
                        bspec_out[i * nfft + j] += v;
                        if (j != i) bspec_out[j * nfft + i] += v;
                    }
                }
            }
        } else {
            /* MATLAB BISPECDx: (i,j) = Xf[i] * Yf[j] * conj(Zf[ij]) */
            SP_OMP_PARALLEL_FOR
            for (long long ii = 0; ii < (long long)nfft; ii++) {
                size_t i = (size_t)ii;
                sp_c128 xi = Xf[i];
                sp_c128 *rowp = bspec_out + i * nfft;
                size_t split = nfft - i;
                for (size_t j = 0; j < split; j++)
                    rowp[j] += xi * Yf[j] * conj(Zf[i + j]);
                for (size_t j = split; j < nfft; j++)
                    rowp[j] += xi * Yf[j] * conj(Zf[i + j - nfft]);
            }
        }
    }

    /* fftshift + average */
    fftshift2d(bspec_out, nfft);
    for (size_t i = 0; i < nfft * nfft; i++) bspec_out[i] /= (double)nrecs;

    if (winsize > 1 && opwind_d) {
        conv2d_same(tmp2, bspec_out, opwind_d, nfft, winsize);
        memcpy(bspec_out, tmp2, nfft * nfft * sizeof(sp_c128));
    }
    waxis_fill(waxis_out, nfft);
    st = SP_OK;

done:
    free(opwind_d);
    free(opwind_c);
    free(Xf);
    free(Yf);
    free(Zf);
    free(seg);
    free(tmp2);
    return st;
}

/* ------------------------------------------------------------------ */
/* bispectrum, indirect method                                         */
/* ------------------------------------------------------------------ */

sp_status sp_bispeci(const sp_c128 *y, size_t N, long nlag_in, size_t nsamp,
                     int overlap, int biased, size_t nfft, int wind_mode,
                     sp_c128 *bspec_out, double *waxis_out) {
    long nlag = nlag_in;
    size_t nlags, ov, nadv, nrecs;
    sp_c128 *c3 = NULL, *cmat = NULL, *wcmat = NULL, *seg = NULL;
    double *window = NULL;
    sp_status st;

    if (!y || !bspec_out || !waxis_out) return SP_ERR_ARG;
    if (N == 0) return SP_ERR_ARG;
    if (nsamp == 0 || nsamp > N) nsamp = N;
    nlag = nlag < (long)nsamp - 1 ? nlag : (long)nsamp - 1;
    if (nlag < 1) return SP_ERR_ARG;
    if (nfft < (size_t)(2 * nlag + 1))
        nfft = (size_t)pow(2.0, ceil(log2((double)nsamp)));
    if (nfft < (size_t)(2 * nlag + 1)) return SP_ERR_ARG;

    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
    nadv = nsamp - ov;
    if (nadv == 0) return SP_ERR_ARG;
    nrecs = (size_t)sp_fix((double)(N - ov) / (double)nadv);
    if (nrecs == 0) return SP_ERR_ARG;

    nlags = (size_t)(2 * nlag + 1);

    /* lag window (Parzen-style or hexagonal unity) */
    window = (double *)calloc(nlags, sizeof(double));
    c3 = (sp_c128 *)calloc((size_t)(nlag + 1) * (size_t)(nlag + 1),
                           sizeof(sp_c128));
    cmat = (sp_c128 *)calloc(nlags * nlags, sizeof(sp_c128));
    wcmat = (sp_c128 *)calloc(nfft * nfft, sizeof(sp_c128));
    seg = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
    if (!window || !c3 || !cmat || !wcmat || !seg) {
        free(window);
        free(c3);
        free(cmat);
        free(wcmat);
        free(seg);
        return SP_ERR_MEM;
    }
    if (wind_mode == 0) {
        window[0] = 1.0;
        for (long i = 1; i <= nlag; i++) {
            double t = M_PI * (double)i / (double)nlag;
            window[i] = sin(t) / t;
        }
    } else {
        for (long i = 0; i <= nlag; i++) window[i] = 1.0;
    }

    /* 3rd-order cumulants in the non-redundant quadrant:
     * c3(i,j) = sum_n conj(x(n) x(n+j)) x(n+i), i >= j */
    for (size_t k = 0; k < nrecs; k++) {
        size_t base = k * nadv;
        if (base + nsamp > N) {
            st = SP_ERR_SHAPE;
            goto done;
        }
        memcpy(seg, y + base, nsamp * sizeof(sp_c128));
        sp_c128 m = vec_mean(seg, nsamp);
        for (size_t n = 0; n < nsamp; n++) seg[n] -= m;
        for (long j = 0; j <= nlag; j++) {
            for (long i = j; i <= nlag; i++) {
                sp_c128 acc = 0.0;
                for (size_t n = 0; n + (size_t)i < nsamp; n++)
                    acc += conj(seg[n] * seg[n + (size_t)j]) *
                           seg[n + (size_t)i];
                if (biased) acc /= (double)nsamp;
                else acc /= (double)(nsamp - (size_t)i);
                c3[(size_t)i * (size_t)(nlag + 1) + (size_t)j] += acc;
            }
        }
    }
    for (size_t i = 0; i < (size_t)(nlag + 1) * (size_t)(nlag + 1); i++)
        c3[i] /= (double)nrecs;

    /* complete the first quadrant: c3 = c3 + tril(c3, -1)' */
    for (long i = 1; i <= nlag; i++)
        for (long j = 0; j < i; j++)
            c3[(size_t)j * (size_t)(nlag + 1) + (size_t)i] +=
                c3[(size_t)i * (size_t)(nlag + 1) + (size_t)j];

    /* assemble the full (2*nlag+1)^2 cmat exactly as the Python reference:
     * cmat = [[c33, c32, 0], [[c34; 0], c3]] */
    {
        long nL = nlag;
        sp_c128 *c31 = (sp_c128 *)calloc((size_t)nL * (size_t)nL, sizeof(sp_c128));
        sp_c128 *c32 = (sp_c128 *)calloc((size_t)nL * (size_t)nL, sizeof(sp_c128));
        sp_c128 *c33 = (sp_c128 *)calloc((size_t)nL * (size_t)nL, sizeof(sp_c128));
        sp_c128 *c34 = (sp_c128 *)calloc((size_t)nL * (size_t)nL, sizeof(sp_c128));
        if (!c31 || !c32 || !c33 || !c34) {
            free(c31);
            free(c32);
            free(c33);
            free(c34);
            st = SP_ERR_MEM;
            goto done;
        }
        for (long i = 0; i < nL; i++)
            for (long j = 0; j < nL; j++)
                c31[(size_t)i * (size_t)nL + (size_t)j] =
                    c3[(size_t)(i + 1) * (size_t)(nlag + 1) + (size_t)(j + 1)];
        for (long i = 0; i < nL; i++) {
            long L = nL - i;
            for (long t = 0; t < L; t++) {
                sp_c128 xv = c31[(size_t)(i + t) * (size_t)nL + (size_t)i];
                c32[(size_t)(nL - 1 - i) * (size_t)nL + (size_t)t] = xv;
                c34[(size_t)t * (size_t)nL + (size_t)(nL - 1 - i)] = xv;
            }
            if (i + 1 < nL) {
                /* x2 = flipud(x[1:]); c33 += diag(x2, +/-(i+1)) */
                for (long sb = 0; sb < L - 1; sb++) {
                    sp_c128 xv = c31[(size_t)(i + (L - 1 - sb)) * (size_t)nL +
                                    (size_t)i];
                    if (sb + i + 1 < nL) {
                        c33[(size_t)sb * (size_t)nL + (size_t)(sb + i + 1)] += xv;
                        c33[(size_t)(sb + i + 1) * (size_t)nL + (size_t)sb] += xv;
                    }
                }
            }
        }
        /* c33 += diag(c3[0, nlag:0:-1]) */
        for (long t = 0; t < nL; t++)
            c33[(size_t)t * (size_t)nL + (size_t)t] +=
                c3[0 + (size_t)(nL - t)];
        /* block assembly: c33 | c32 | 0 ; c34 | c3 (c3 keeps its extra
         * last row/col) */
        for (long i = 0; i < nL; i++) {
            for (long j = 0; j < nL; j++) {
                cmat[(size_t)i * nlags + (size_t)j] = c33[(size_t)i * (size_t)nL + (size_t)j];
                cmat[(size_t)i * nlags + (size_t)(nL + j)] = c32[(size_t)i * (size_t)nL + (size_t)j];
                cmat[(size_t)(nL + i) * nlags + (size_t)j] = c34[(size_t)i * (size_t)nL + (size_t)j];
                cmat[(size_t)(nL + i) * nlags + (size_t)(nL + j)] = c3[(size_t)i * (size_t)(nlag + 1) + (size_t)j];
            }
        }
        /* row 2*nlag of the c34 block is zeros; col 2*nlag of the top block
         * is zeros; c3's last row/col fills (2*nlag, nL..2*nlag) and
         * (nL..2*nlag, 2*nlag) */
        for (long j = 0; j <= nlag; j++) {
            cmat[(size_t)(2 * nL) * nlags + (size_t)(nL + j)] =
                c3[(size_t)nL * (size_t)(nlag + 1) + (size_t)j];
            cmat[(size_t)(nL + j) * nlags + (size_t)(2 * nL)] =
                c3[(size_t)j * (size_t)(nlag + 1) + (size_t)nL];
        }
        cmat[(size_t)(2 * nL) * nlags + (size_t)(2 * nL)] =
            c3[(size_t)nL * (size_t)(nlag + 1) + (size_t)nL];
        free(c31);
        free(c32);
        free(c33);
        free(c34);
    }

    /* lag-domain windowing: wcmat(:,k) = cmat(:,k) * win(|i-k|) win(|i|) win(|k|) */
    if (wind_mode != -1) {
        for (long k = -nlag; k <= nlag; k++) {
            for (long i = -nlag; i <= nlag; i++) {
                long wk = k - (-nlag); /* column index */
                long wi = i - (-nlag);
                double w1 = window[(size_t)labs(i - k)];
                double w2 = window[(size_t)labs(i)];
                double w3 = window[(size_t)labs(k)];
                wcmat[(size_t)wi * nfft + (size_t)wk] =
                    cmat[(size_t)wi * nlags + (size_t)wk] * (w1 * w2 * w3);
            }
        }
    } else {
        for (size_t i = 0; i < nlags; i++)
            memcpy(&wcmat[i * nfft], &cmat[i * nlags], nlags * sizeof(sp_c128));
    }

    /* 2-D FFT with zero padding to nfft x nfft (rows then columns) */
    for (size_t r = 0; r < nlags; r++) {
        if (sp_fft_forward(&wcmat[r * nfft], nfft) != 0) {
            st = SP_ERR_INTERNAL;
            goto done;
        }
    }
    {
        sp_c128 *col = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
        if (!col) {
            st = SP_ERR_MEM;
            goto done;
        }
        for (size_t c = 0; c < nfft; c++) {
            for (size_t r = 0; r < nfft; r++)
                col[r] = (r < nlags) ? wcmat[r * nfft + c] : 0.0;
            if (sp_fft_forward(col, nfft) != 0) {
                free(col);
                st = SP_ERR_INTERNAL;
                goto done;
            }
            for (size_t r = 0; r < nfft; r++) wcmat[r * nfft + c] = col[r];
        }
        free(col);
    }
    memcpy(bspec_out, wcmat, nfft * nfft * sizeof(sp_c128));
    fftshift2d(bspec_out, nfft);
    waxis_fill(waxis_out, nfft);
    st = SP_OK;

done:
    free(window);
    free(c3);
    free(cmat);
    free(wcmat);
    free(seg);
    return st;
}

/* ------------------------------------------------------------------ */
/* trispectrum / tricoherence (reduced-grid direct method)             */
/* ------------------------------------------------------------------ */

sp_status sp_trispectrum(const sp_c128 *y, size_t N, size_t nsamp, size_t nfft,
                         int window_type, int overlap, sp_c128 *tspec_out,
                         double *waxis_out, size_t *mfi_out) {
    size_t ov, nadv, nrecord, mfi, nf;
    double *wind = NULL;
    sp_c128 *seg = NULL, *X = NULL;

    if (!y || !tspec_out || !waxis_out || !mfi_out) return SP_ERR_ARG;
    if (nsamp == 0 || nsamp > N) nsamp = N;
    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
    nadv = nsamp - ov;
    if (nadv == 0) return SP_ERR_ARG;
    nrecord = (size_t)sp_fix((double)(N - ov) / (double)nadv);
    if (nrecord == 0) return SP_ERR_ARG;
    nf = nfft / 2 + 1;
    mfi = nf < nfft / 8 ? nf : nfft / 8;
    if (mfi == 0) return SP_ERR_ARG;
    for (size_t i = 0; i < mfi * mfi * mfi; i++) tspec_out[i] = 0.0;

    wind = (double *)malloc(nsamp * sizeof(double));
    seg = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
    X = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    if (!wind || !seg || !X) {
        free(wind);
        free(seg);
        free(X);
        return SP_ERR_MEM;
    }
    switch (window_type) {
        case 1: build_hann(wind, nsamp); break;
        case 2: build_hamming(wind, nsamp); break;
        case 3: build_blackman(wind, nsamp); break;
        default:
            for (size_t n = 0; n < nsamp; n++) wind[n] = 1.0;
    }
    /* normalize: wind / ||wind|| * sqrt(nsamp) */
    {
        double nrm = 0.0;
        for (size_t n = 0; n < nsamp; n++) nrm += wind[n] * wind[n];
        nrm = sqrt(nrm);
        for (size_t n = 0; n < nsamp; n++)
            wind[n] = nrm > 0.0 ? wind[n] / nrm * sqrt((double)nsamp) : 0.0;
    }

    for (size_t r = 0; r < nrecord; r++) {
        size_t base = r * nadv;
        if (base + nsamp > N) {
            free(wind);
            free(seg);
            free(X);
            return SP_ERR_SHAPE;
        }
        memcpy(seg, y + base, nsamp * sizeof(sp_c128));
        sp_c128 m = vec_mean(seg, nsamp);
        memset(X, 0, nfft * sizeof(sp_c128));
        size_t ncopy = nsamp < nfft ? nsamp : nfft; /* numpy fft(x, n): first n pts */
        for (size_t n = 0; n < ncopy; n++)
            X[n] = (seg[n] - m) * wind[n];
        if (sp_fft_forward(X, nfft) != 0) {
            free(wind);
            free(seg);
            free(X);
            return SP_ERR_INTERNAL;
        }
        for (size_t k1 = 0; k1 < mfi; k1++) {
            for (size_t k2 = 0; k2 < mfi; k2++) {
                for (size_t k3 = 0; k3 < mfi; k3++) {
                    size_t k4 = k1 + k2 + k3;
                    if (k4 >= mfi) break;
                    tspec_out[(k1 * mfi + k2) * mfi + k3] +=
                        X[k1] * X[k2] * X[k3] * conj(X[k4]);
                }
            }
        }
    }
    for (size_t i = 0; i < mfi * mfi * mfi; i++)
        tspec_out[i] /= (double)nrecord;

    for (size_t i = 0; i < mfi; i++)
        waxis_out[i] = (double)i / (double)nfft;
    *mfi_out = mfi;
    free(wind);
    free(seg);
    free(X);
    return SP_OK;
}

sp_status sp_tricoherence(const sp_c128 *y, size_t N, size_t nsamp, size_t nfft,
                          int window_type, int overlap, double *tricoh_out,
                          double *waxis_out, size_t *mfi_out) {
    size_t mfi;
    sp_c128 *T = NULL;
    double *P = NULL;
    sp_status st;
    size_t ov, nadv, nrecord;
    double *wind = NULL;
    sp_c128 *seg = NULL, *X = NULL;

    if (nfft == 0) return SP_ERR_ARG;
    mfi = (nfft / 2 + 1) < (nfft / 8) ? (nfft / 2 + 1) : (nfft / 8);
    T = (sp_c128 *)calloc(mfi * mfi * mfi, sizeof(sp_c128));
    P = (double *)calloc(mfi, sizeof(double));
    if (!T || !P) {
        free(T);
        free(P);
        return SP_ERR_MEM;
    }
    st = sp_trispectrum(y, N, nsamp, nfft, window_type, overlap, T, waxis_out,
                        mfi_out);
    if (st != SP_OK) {
        free(T);
        free(P);
        return st;
    }
    /* recompute the averaged periodogram with identical segmentation */
    if (nsamp == 0 || nsamp > N) nsamp = N;
    if (overlap < 0) overlap = 0;
    if (overlap > 99) overlap = 99;
    ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
    nadv = nsamp - ov;
    nrecord = (size_t)sp_fix((double)(N - ov) / (double)nadv);
    if (nrecord == 0 || nadv == 0) {
        free(T);
        free(P);
        return SP_ERR_ARG;
    }
    wind = (double *)malloc(nsamp * sizeof(double));
    seg = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
    X = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    if (!wind || !seg || !X) {
        free(wind);
        free(seg);
        free(X);
        free(T);
        free(P);
        return SP_ERR_MEM;
    }
    switch (window_type) {
        case 1: build_hann(wind, nsamp); break;
        case 2: build_hamming(wind, nsamp); break;
        case 3: build_blackman(wind, nsamp); break;
        default:
            for (size_t n = 0; n < nsamp; n++) wind[n] = 1.0;
    }
    {
        double nrm = 0.0;
        for (size_t n = 0; n < nsamp; n++) nrm += wind[n] * wind[n];
        nrm = sqrt(nrm);
        for (size_t n = 0; n < nsamp; n++)
            wind[n] = nrm > 0.0 ? wind[n] / nrm * sqrt((double)nsamp) : 0.0;
    }
    for (size_t r = 0; r < nrecord; r++) {
        size_t base = r * nadv;
        memcpy(seg, y + base, nsamp * sizeof(sp_c128));
        sp_c128 m = vec_mean(seg, nsamp);
        memset(X, 0, nfft * sizeof(sp_c128));
        {
            size_t ncopy = nsamp < nfft ? nsamp : nfft;
            for (size_t n = 0; n < ncopy; n++) X[n] = (seg[n] - m) * wind[n];
        }
        if (sp_fft_forward(X, nfft) != 0) {
            free(wind);
            free(seg);
            free(X);
            free(T);
            free(P);
            return SP_ERR_INTERNAL;
        }
        for (size_t k = 0; k < mfi; k++) P[k] += cabs(X[k]) * cabs(X[k]);
    }
    for (size_t k = 0; k < mfi; k++) P[k] /= (double)nrecord;

    for (size_t i = 0; i < mfi * mfi * mfi; i++) tricoh_out[i] = 0.0;
    for (size_t k1 = 0; k1 < mfi; k1++)
        for (size_t k2 = 0; k2 < mfi; k2++)
            for (size_t k3 = 0; k3 < mfi; k3++) {
                size_t k4 = k1 + k2 + k3;
                if (k4 >= mfi) break;
                double denom = sqrt(P[k1] * P[k2] * P[k3] * P[k4]);
                double v = 0.0;
                if (denom > 0.0) {
                    v = cabs(T[(k1 * mfi + k2) * mfi + k3]) / denom;
                    if (v > 1.0) v = 1.0;
                }
                tricoh_out[(k1 * mfi + k2) * mfi + k3] = v;
            }
    free(wind);
    free(seg);
    free(X);
    free(T);
    free(P);
    return SP_OK;
}


/* ------------------------------------------------------------------ */
/* HOSA-style full trispectrum (direct method, wraparound cube)        */
/* ------------------------------------------------------------------ */

sp_status sp_trispec(const sp_c128 *y, size_t N, size_t cols, size_t nsamp,
                     size_t nfft, int overlap, sp_c128 *t_out,
                     double *waxis_out) {
    size_t ov, nadv, nrecs;
    sp_c128 *X = NULL, *seg = NULL;
    sp_status st;

    if (!y || !t_out || !waxis_out) return SP_ERR_ARG;
    if (nfft > 512) return SP_ERR_ARG; /* nfft^3 memory guard */
    st = seg_counts(N / (cols ? cols : 1), cols, nsamp, overlap, &ov, &nadv,
                    &nrecs);
    if (st != SP_OK) return st;

    memset(t_out, 0, nfft * nfft * nfft * sizeof(sp_c128));

    X = (sp_c128 *)malloc(nfft * sizeof(sp_c128));
    seg = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
    if (!X || !seg) {
        free(X);
        free(seg);
        return SP_ERR_MEM;
    }

    for (size_t k = 0; k < nrecs; k++) {
        size_t base = k * nadv;
        if (base + nsamp > N) {
            free(X);
            free(seg);
            return SP_ERR_SHAPE;
        }
        memcpy(seg, y + base, nsamp * sizeof(sp_c128));
        sp_c128 m = vec_mean(seg, nsamp);
        memset(X, 0, nfft * sizeof(sp_c128));
        size_t ncopy = nsamp < nfft ? nsamp : nfft;
        for (size_t n = 0; n < ncopy; n++) X[n] = seg[n] - m;
        if (sp_fft_forward(X, nfft) != 0) {
            free(X);
            free(seg);
            return SP_ERR_INTERNAL;
        }
        for (size_t n = 0; n < nfft; n++) X[n] /= (double)nsamp;

        SP_OMP_PARALLEL_FOR
        for (long long kk1 = 0; kk1 < (long long)nfft; kk1++) {
            size_t k1 = (size_t)kk1;
            sp_c128 a = X[k1];
            for (size_t k2 = 0; k2 < nfft; k2++) {
                sp_c128 ab = a * X[k2];
                size_t base_ij = (k1 * nfft + k2) * nfft;
                size_t k12 = (k1 + k2) % nfft;
                size_t split = nfft - k12;
                for (size_t k3 = 0; k3 < split; k3++)
                    t_out[base_ij + k3] += ab * X[k3] * conj(X[k12 + k3]);
                for (size_t k3 = split; k3 < nfft; k3++)
                    t_out[base_ij + k3] += ab * X[k3] * conj(X[k12 + k3 - nfft]);
            }
        }
    }

    double inv = 1.0 / (double)nrecs;
    for (size_t i = 0; i < nfft * nfft * nfft; i++) t_out[i] *= inv;
    for (size_t i = 0; i < nfft; i++) waxis_out[i] = (double)i / (double)nfft;

    free(X);
    free(seg);
    return SP_OK;
}
