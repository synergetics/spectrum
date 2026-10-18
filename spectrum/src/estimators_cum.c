/* estimators_cum.c - cumulant estimator kernels, exact ports of the corrected
 * Python reference implementations (spectrum/cum*est.py, spectrum/cum*x.py)
 * which themselves follow the HOSA MATLAB toolbox semantics.
 *
 * Segment extraction for multi-column inputs mirrors numpy's
 * x[ind].ravel(order="F") where ind = base + arange(nsamp) selects rows.
 * With a Fortran-order flat buffer of shape (N, cols):
 *   seg[j*nsamp + n] = flat[j*N + base + n]
 */
#include "estimators_cum.h"

static void extract_seg(sp_c128 *dst, const sp_c128 *flat, size_t N,
                        size_t cols, size_t base, size_t nsamp) {
    if (cols == 1) {
        memcpy(dst, flat + base, nsamp * sizeof(sp_c128));
        return;
    }
    for (size_t j = 0; j < cols; j++) {
        for (size_t n = 0; n < nsamp; n++) {
            dst[j * nsamp + n] = flat[j * N + base + n];
        }
    }
}

static sp_c128 seg_mean(const sp_c128 *v, size_t n) {
    /* n as complex accumulation mirrors numpy mean on complex data */
    double sr = 0.0, si = 0.0;
    for (size_t i = 0; i < n; i++) {
        sr += creal(v[i]);
        si += cimag(v[i]);
    }
    return (sr / (double)n) + (si / (double)n) * I;
}

static void mean_remove(sp_c128 *v, size_t n) {
    sp_c128 m = seg_mean(v, n);
    for (size_t i = 0; i < n; i++) v[i] -= m;
}

/* ------------------------------------------------------------------ */
/* cum2est: C2(m) = E[x*(n) x(n+m)], biased: /nsamp, unbiased: /(nsamp-m) */
sp_status sp_cum2est(const sp_c128 *y, size_t N, int maxlag, size_t nsamp,
                     int overlap, int biased, sp_c128 *out) {
    if (!y || !out || maxlag < 0 || nsamp == 0) return SP_ERR_ARG;
    {
        size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
        size_t nadv = nsamp - ov;
        long nrecord;
        sp_c128 *pos;
        if (nadv == 0 || N < ov) return SP_ERR_ARG;
        nrecord = sp_fix((double)(N - ov) / (double)nadv);
        if (nrecord < 1) return SP_ERR_ARG;

        pos = (sp_c128 *)malloc((size_t)(maxlag + 1) * sizeof(sp_c128));
        if (!pos) return SP_ERR_MEM;
        for (int k = 0; k <= maxlag; k++) pos[k] = 0.0;

        sp_c128 *x = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        if (!x) {
            free(pos);
            return SP_ERR_MEM;
        }

        for (long r = 0; r < nrecord; r++) {
            size_t base = (size_t)r * nadv;
            if (base + nsamp > N) {
                free(x);
                free(pos);
                return SP_ERR_SHAPE;
            }
            memcpy(x, y + base, nsamp * sizeof(sp_c128));
            mean_remove(x, nsamp);
            for (int k = 0; k <= maxlag; k++) {
                sp_c128 acc = 0.0;
                for (size_t n = 0; n + (size_t)k < nsamp; n++) {
                    acc += conj(x[n]) * x[n + (size_t)k];
                }
                pos[k] += acc;
            }
        }
        free(x);

        if (biased) {
            double s = (double)nsamp * (double)nrecord;
            for (int k = 0; k <= maxlag; k++) pos[k] /= s;
        } else {
            for (int k = 0; k <= maxlag; k++)
                pos[k] /= (double)nrecord * (double)(nsamp - (size_t)k);
        }

        if (maxlag > 0) {
            for (int i = 0; i < maxlag; i++) out[i] = conj(pos[maxlag - i]);
            for (int k = 0; k <= maxlag; k++) out[maxlag + k] = pos[k];
        } else {
            out[0] = pos[0];
        }
        free(pos);
        return SP_OK;
    }
}

/* ------------------------------------------------------------------ */
/* cum2x: C(m) = E[x*(n) y(n+m)]                                       */
sp_status sp_cum2x(const sp_c128 *x, const sp_c128 *y, size_t N, size_t cols,
                   int maxlag, size_t nsamp, int overlap, int biased,
                   sp_c128 *out) {
    if (!x || !y || !out || maxlag < 0) return SP_ERR_ARG;
    {
        size_t lx = N / (cols ? cols : 1);
        size_t nrecs = cols ? cols : 1;
        size_t nlags = (size_t)(2 * maxlag + 1);
        int zlag = maxlag;

        if (lx == 1) {
            lx = nrecs;
            nrecs = 1;
        }
        if (nrecs > 1) nsamp = lx;
        if (nsamp == 0 || nsamp > lx) nsamp = lx;
        if (nrecs > 1) overlap = 0;
        if (overlap < 0) overlap = 0;
        if (overlap > 99) overlap = 99;

        size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
        size_t nadv = nsamp - ov;
        if (nrecs == 1)
            nrecs = (size_t)sp_fix((double)(lx - ov) / (double)nadv);

        sp_c128 *yc = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        sp_c128 *xs = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        sp_c128 *ys = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        if (!yc || !xs || !ys) {
            free(yc);
            free(xs);
            free(ys);
            return SP_ERR_MEM;
        }

        for (size_t k = 0; k < nrecs; k++) {
            size_t base = k * nadv;
            if (base + nsamp > N) {
                free(yc);
                free(xs);
                free(ys);
                return SP_ERR_SHAPE;
            }
            extract_seg(xs, x, N, cols, base, nsamp);
            extract_seg(ys, y, N, cols, base, nsamp);
            mean_remove(xs, nsamp);
            mean_remove(ys, nsamp);

            sp_c128 acc = 0.0;
            for (size_t n = 0; n < nsamp; n++) acc += conj(xs[n]) * ys[n];
            yc[zlag] += acc;

            for (int m = 1; m <= maxlag; m++) {
                sp_c128 a = 0.0, b = 0.0;
                for (size_t n = (size_t)m; n < nsamp; n++)
                    a += conj(xs[n]) * ys[n - (size_t)m];
                for (size_t n = 0; n + (size_t)m < nsamp; n++)
                    b += conj(xs[n]) * ys[n + (size_t)m];
                yc[zlag - m] += a;
                yc[zlag + m] += b;
            }
        }
        free(xs);
        free(ys);

        for (size_t i = 0; i < nlags; i++) {
            int lag = (int)i - maxlag;
            double denom = biased ? (double)nsamp
                                  : (double)(lx - (size_t)abs(lag));
            yc[i] = yc[i] / denom / (double)nrecs;
        }
        memcpy(out, yc, nlags * sizeof(sp_c128));
        free(yc);
        return SP_OK;
    }
}

/* ------------------------------------------------------------------ */
/* cum3est: C3(m, k1) via the IV construction                         */
sp_status sp_cum3est(const sp_c128 *y, size_t N, int maxlag, size_t nsamp,
                     int overlap, int biased, int k1, sp_c128 *out) {
    if (!y || !out || maxlag < 0 || nsamp == 0) return SP_ERR_ARG;
    {
        int minlag = -maxlag;
        size_t nlags = (size_t)(2 * maxlag + 1);
        int zlag = maxlag;
        size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
        size_t nadv = nsamp - ov;
        long nrecord;
        sp_c128 *z, *x;
        double lsamp;

        if (nadv == 0) return SP_ERR_ARG;
        nrecord = sp_fix((double)(N - ov) / (double)nadv);
        if (nrecord < 1) return SP_ERR_ARG;

        z = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        x = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        if (!z || !x) {
            free(z);
            free(x);
            return SP_ERR_MEM;
        }

        for (long r = 0; r < nrecord; r++) {
            size_t base = (size_t)r * nadv;
            if (base + nsamp > N) {
                free(z);
                free(x);
                return SP_ERR_SHAPE;
            }
            memcpy(x, y + base, nsamp * sizeof(sp_c128));
            mean_remove(x, nsamp);

            sp_c128 *iv = (sp_c128 *)calloc(nsamp, sizeof(sp_c128));
            if (!iv) {
                free(z);
                free(x);
                return SP_ERR_MEM;
            }
            if (k1 > 0) {
                for (size_t n = 0; n + (size_t)k1 < nsamp; n++)
                    iv[n] = x[n] * conj(x[n + (size_t)k1]);
            } else {
                for (size_t n = (size_t)(-k1); n < nsamp; n++)
                    iv[n] = x[n] * conj(x[n + (size_t)k1]);
            }

            sp_c128 acc = 0.0;
            for (size_t n = 0; n < nsamp; n++) acc += conj(iv[n]) * x[n];
            z[zlag] += acc;
            for (int m = 1; m <= maxlag; m++) {
                sp_c128 a = 0.0, b = 0.0;
                for (size_t n = (size_t)m; n < nsamp; n++)
                    a += conj(iv[n]) * x[n - (size_t)m];
                for (size_t n = 0; n + (size_t)m < nsamp; n++)
                    b += conj(iv[n]) * x[n + (size_t)m];
                z[zlag - m] += a;
                z[zlag + m] += b;
            }
            free(iv);
        }
        free(x);

        lsamp = biased ? 0.0 : (double)(nsamp - (size_t)abs(k1));
        for (size_t i = 0; i < nlags; i++) {
            int lag = (int)i + minlag;
            if (biased) {
                z[i] /= (double)nsamp * (double)nrecord;
            } else {
                z[i] /= ((lsamp - (double)abs(lag)) * (double)nrecord);
            }
        }
        memcpy(out, z, nlags * sizeof(sp_c128));
        free(z);
        return SP_OK;
    }
}

/* ------------------------------------------------------------------ */
/* cum3x: C(m, k1) = E[x*(n) y(n+m) z*(n+k1)]                          */
sp_status sp_cum3x(const sp_c128 *x, const sp_c128 *y, const sp_c128 *z,
                   size_t N, size_t cols, int maxlag, size_t nsamp,
                   int overlap, int biased, int k1, sp_c128 *out) {
    if (!x || !y || !z || !out || maxlag < 0) return SP_ERR_ARG;
    {
        size_t lx = N / (cols ? cols : 1);
        size_t nrecs = cols ? cols : 1;
        size_t nlags = (size_t)(2 * maxlag + 1);
        int zlag = maxlag;

        if (lx == 1) {
            lx = nrecs;
            nrecs = 1;
        }
        if (nrecs > 1) nsamp = lx;
        if (nsamp == 0 || nsamp > lx) nsamp = lx;
        if (nrecs > 1) overlap = 0;
        if (overlap < 0) overlap = 0;
        if (overlap > 99) overlap = 99;

        size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
        size_t nadv = nsamp - ov;
        if (nrecs == 1)
            nrecs = (size_t)sp_fix((double)(lx - ov) / (double)nadv);
        if (nadv == 0) return SP_ERR_ARG;

        double lsamp = biased ? 0.0 : (double)(lx - (size_t)abs(k1));

        sp_c128 *yc = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        sp_c128 *xs = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        sp_c128 *ys = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        sp_c128 *zs = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        sp_c128 *u = (sp_c128 *)calloc(nsamp, sizeof(sp_c128));
        if (!yc || !xs || !ys || !zs || !u) {
            free(yc);
            free(xs);
            free(ys);
            free(zs);
            free(u);
            return SP_ERR_MEM;
        }

        size_t ind_lo, ind_hi; /* u[n] = xs[n] * zs[n + k1] for n in [lo,hi) */
        if (k1 >= 0) {
            ind_lo = 0;
            ind_hi = nsamp - (size_t)k1;
        } else {
            ind_lo = (size_t)(-k1);
            ind_hi = nsamp;
        }

        for (size_t k = 0; k < nrecs; k++) {
            size_t base = k * nadv;
            if (base + nsamp > N) {
                free(yc);
                free(xs);
                free(ys);
                free(zs);
                free(u);
                return SP_ERR_SHAPE;
            }
            extract_seg(xs, x, N, cols, base, nsamp);
            extract_seg(ys, y, N, cols, base, nsamp);
            extract_seg(zs, z, N, cols, base, nsamp);
            mean_remove(xs, nsamp);
            mean_remove(ys, nsamp);
            mean_remove(zs, nsamp);
            for (size_t n = 0; n < nsamp; n++) zs[n] = conj(zs[n]);

            for (size_t n = 0; n < nsamp; n++) u[n] = 0.0;
            /* u(n) = x(n) z*(n+k1); zs is already conjugated */
            if (k1 >= 0) {
                for (size_t n = 0; n + (size_t)k1 < nsamp; n++)
                    u[n] = xs[n] * zs[n + (size_t)k1];
            } else {
                for (size_t n = (size_t)(-k1); n < nsamp; n++)
                    u[n] = xs[n] * zs[n + (size_t)k1];
            }

            sp_c128 acc = 0.0;
            for (size_t n = 0; n < nsamp; n++) acc += conj(u[n]) * ys[n];
            yc[zlag] += acc;
            for (int m = 1; m <= maxlag; m++) {
                sp_c128 a = 0.0, b = 0.0;
                for (size_t n = (size_t)m; n < nsamp; n++)
                    a += conj(u[n]) * ys[n - (size_t)m];
                for (size_t n = 0; n + (size_t)m < nsamp; n++)
                    b += conj(u[n]) * ys[n + (size_t)m];
                yc[zlag - m] += a;
                yc[zlag + m] += b;
            }
        }
        free(xs);
        free(ys);
        free(zs);
        free(u);

        for (size_t i = 0; i < nlags; i++) {
            int lag = (int)i - maxlag;
            double denom = biased ? (double)nsamp
                                  : (lsamp - (double)abs(lag));
            yc[i] = yc[i] / denom / (double)nrecs;
        }
        memcpy(out, yc, nlags * sizeof(sp_c128));
        free(yc);
        return SP_OK;
    }
}

/* ------------------------------------------------------------------ */
/* cum4est: C4(m, k1, k2) via IV chain and moment products              */
sp_status sp_cum4est(const sp_c128 *y, size_t N, int maxlag, size_t nsamp,
                     int overlap, int biased, int k1, int k2, sp_c128 *out) {
    if (!y || !out || maxlag < 0 || nsamp == 0) return SP_ERR_ARG;
    {
        size_t nlags = (size_t)(2 * maxlag + 1);
        int zlag = maxlag;
        int mlag = maxlag + (abs(k1) > abs(k2) ? abs(k1) : abs(k2));
        int t = abs(k1 - k2);
        int mlag1, nlag = maxlag;
        long nrecord;
        size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
        size_t nadv = nsamp - ov;
        int kmin = (0 < k1 ? 0 : k1) < k2 ? (0 < k1 ? 0 : k1) : k2;
        int kmax = (0 > k1 ? 0 : k1) > k2 ? (0 > k1 ? 0 : k1) : k2;
        sp_c128 *y_cum, *tmp, *x, *R_yy, *M_yy, *scale;
        int complex_flag = 0;

        if (mlag < t) mlag = t;
        mlag1 = mlag + 1;
        if (kmin > 0) kmin = 0;
        if (kmax < 0) kmax = 0;
        if (nadv == 0) return SP_ERR_ARG;
        nrecord = sp_fix((double)(N - ov) / (double)nadv);
        if (nrecord < 1) return SP_ERR_ARG;

        for (size_t i = 0; i < N; i++) {
            if (cimag(y[i]) != 0.0) {
                complex_flag = 1;
                break;
            }
        }

        y_cum = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        tmp = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        x = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        R_yy = (sp_c128 *)malloc((size_t)(2 * mlag + 1) * sizeof(sp_c128));
        M_yy = (sp_c128 *)malloc((size_t)(2 * mlag + 1) * sizeof(sp_c128));
        scale = (sp_c128 *)malloc(nlags * sizeof(sp_c128));
        if (!y_cum || !tmp || !x || !R_yy || !M_yy || !scale) {
            free(y_cum);
            free(tmp);
            free(x);
            free(R_yy);
            free(M_yy);
            free(scale);
            return SP_ERR_MEM;
        }

        if (biased) {
            for (size_t i = 0; i < nlags; i++) scale[i] = 1.0 / (double)nsamp;
        } else {
            for (size_t i = 0; i < nlags; i++) {
                int lag = (int)i - maxlag;
                int mx = lag > kmax ? lag : kmax;
                int mn = lag < kmin ? lag : kmin;
                scale[i] = 1.0 / (double)(nsamp - (size_t)mx + (size_t)mn);
            }
        }

        for (long r = 0; r < nrecord; r++) {
            size_t base = (size_t)r * nadv;
            if (base + nsamp > N) {
                free(y_cum);
                free(tmp);
                free(x);
                free(R_yy);
                free(M_yy);
                free(scale);
                return SP_ERR_SHAPE;
            }
            memcpy(x, y + base, nsamp * sizeof(sp_c128));
            mean_remove(x, nsamp);

            /* IV chain: z(n) = x(n) conj(x(n+k1)) x(n+k2) */
            sp_c128 *z = (sp_c128 *)calloc(nsamp, sizeof(sp_c128));
            if (!z) {
                free(y_cum);
                free(tmp);
                free(x);
                free(R_yy);
                free(M_yy);
                free(scale);
                return SP_ERR_MEM;
            }
            if (k1 >= 0) {
                for (size_t n = 0; n + (size_t)k1 < nsamp; n++)
                    z[n] = x[n] * conj(x[n + (size_t)k1]);
            } else {
                for (size_t n = (size_t)(-k1); n < nsamp; n++)
                    z[n] = x[n] * conj(x[n + (size_t)k1]);
            }
            if (k2 >= 0) {
                for (size_t n = 0; n + (size_t)k2 < nsamp; n++)
                    z[n] *= x[n + (size_t)k2];
            } else {
                for (size_t n = (size_t)(-k2); n < nsamp; n++)
                    z[n] *= x[n + (size_t)k2];
            }

            for (size_t i = 0; i < nlags; i++) tmp[i] = 0.0;
            sp_c128 acc = 0.0;
            for (size_t n = 0; n < nsamp; n++) acc += conj(z[n]) * x[n];
            tmp[zlag] += acc;
            for (int m = 1; m <= maxlag; m++) {
                sp_c128 a = 0.0, b = 0.0;
                for (size_t n = (size_t)m; n < nsamp; n++)
                    a += conj(z[n]) * x[n - (size_t)m];
                for (size_t n = 0; n + (size_t)m < nsamp; n++)
                    b += conj(z[n]) * x[n + (size_t)m];
                tmp[zlag - m] += a;
                tmp[zlag + m] += b;
            }
            for (size_t i = 0; i < nlags; i++) y_cum[i] += tmp[i] * scale[i];
            free(z);

            /* second-order corrections on this segment */
            sp_status st;
            st = sp_cum2est(x, nsamp, mlag, nsamp, 0, biased, R_yy);
            if (st != SP_OK) {
                free(y_cum);
                free(tmp);
                free(x);
                free(R_yy);
                free(M_yy);
                free(scale);
                return st;
            }
            if (complex_flag) {
                sp_c128 *cx = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
                if (!cx) {
                    free(y_cum);
                    free(tmp);
                    free(x);
                    free(R_yy);
                    free(M_yy);
                    free(scale);
                    return SP_ERR_MEM;
                }
                for (size_t n = 0; n < nsamp; n++) cx[n] = conj(x[n]);
                st = sp_cum2x(cx, x, nsamp, 1, mlag, nsamp, 0, biased, M_yy);
                free(cx);
                if (st != SP_OK) {
                    free(y_cum);
                    free(tmp);
                    free(x);
                    free(R_yy);
                    free(M_yy);
                    free(scale);
                    return st;
                }
            } else {
                memcpy(M_yy, R_yy, (size_t)(2 * mlag + 1) * sizeof(sp_c128));
            }

            /* y_cum -= corrections; R_yy index (mlag1+k1-1) is lag k1 etc. */
            for (int m = -maxlag; m <= maxlag; m++) {
                sp_c128 c1 = R_yy[mlag1 + k1 - 1];
                sp_c128 c2 = R_yy[mlag1 - k2 + m - 1];
                sp_c128 c3 = R_yy[k1 - k2 + mlag1 - 1];
                sp_c128 c4 = R_yy[mlag1 + m - 1];
                sp_c128 c5 = M_yy[mlag1 + k2 - 1];
                sp_c128 c6 = M_yy[mlag1 - k1 + m - 1];
                (void)c2;
                y_cum[m + maxlag] -= c1 * c2 + c3 * c4 + c5 * c6;
            }
        }
        free(x);
        free(tmp);
        free(R_yy);
        free(M_yy);
        free(scale);

        for (size_t i = 0; i < nlags; i++) y_cum[i] /= (double)nrecord;
        memcpy(out, y_cum, nlags * sizeof(sp_c128));
        free(y_cum);
        return SP_OK;
    }
}

/* ------------------------------------------------------------------ */
/* cum4x: 4th-order cross-cumulant c4(t1,t2,t3)                        */
sp_status sp_cum4x(const sp_c128 *w, const sp_c128 *x, const sp_c128 *y,
                   const sp_c128 *z, size_t N, size_t cols, int maxlag,
                   size_t nsamp, int overlap, int biased, int k1, int k2,
                   sp_c128 *out) {
    if (!w || !x || !y || !z || !out || maxlag < 0) return SP_ERR_ARG;
    {
        size_t lx = N / (cols ? cols : 1);
        size_t nrecs = cols ? cols : 1;
        size_t nlags = (size_t)(2 * maxlag + 1);
        int zlag = maxlag;
        double sc1, sc2, sc12;
        sp_c128 *y_cum, *tmp, *scale, *ws, *xs, *ys, *zs, *cys, *ziv;
        sp_c128 *R_wx, *R_zx, *M_yx, *cys_buf;

        if (lx == 1) {
            lx = nrecs;
            nrecs = 1;
        }
        if (nrecs > 1) nsamp = lx;
        if (nsamp == 0 || nsamp > lx) nsamp = lx;
        if (nrecs > 1) overlap = 0;
        if (overlap < 0) overlap = 0;
        if (overlap > 99) overlap = 99;

        size_t ov = (size_t)sp_fix((double)overlap / 100.0 * (double)nsamp);
        size_t nadv = nsamp - ov;
        if (nrecs == 1)
            nrecs = (size_t)sp_fix((double)(lx - ov) / (double)nadv);
        if (nadv == 0) return SP_ERR_ARG;

        int kmin = 0 < k1 ? 0 : k1;
        int kmax = 0 > k1 ? 0 : k1;
        if (k2 < kmin) kmin = k2;
        if (k2 > kmax) kmax = k2;

        if (biased) {
            sc1 = sc2 = sc12 = 1.0 / (double)nsamp;
        } else {
            sc1 = 1.0 / (double)(nsamp - (size_t)abs(k1));
            sc2 = 1.0 / (double)(nsamp - (size_t)abs(k2));
            sc12 = 1.0 / (double)(nsamp - (size_t)abs(k1 - k2));
        }

        y_cum = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        tmp = (sp_c128 *)calloc(nlags, sizeof(sp_c128));
        scale = (sp_c128 *)malloc(nlags * sizeof(sp_c128));
        ws = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        xs = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        ys = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        zs = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        cys_buf = (sp_c128 *)malloc(nsamp * sizeof(sp_c128));
        ziv = (sp_c128 *)calloc(nsamp, sizeof(sp_c128));
        R_wx = (sp_c128 *)malloc(nlags * sizeof(sp_c128));
        R_zx = (sp_c128 *)malloc((size_t)(2 * (maxlag + abs(k2)) + 1) * sizeof(sp_c128));
        M_yx = (sp_c128 *)malloc((size_t)(2 * (maxlag + abs(k1)) + 1) * sizeof(sp_c128));
        if (!y_cum || !tmp || !scale || !ws || !xs || !ys || !zs || !cys_buf ||
            !ziv || !R_wx || !R_zx || !M_yx) {
            free(y_cum);
            free(tmp);
            free(scale);
            free(ws);
            free(xs);
            free(ys);
            free(zs);
            free(cys_buf);
            free(ziv);
            free(R_wx);
            free(R_zx);
            free(M_yx);
            return SP_ERR_MEM;
        }

        if (biased) {
            for (size_t i = 0; i < nlags; i++) scale[i] = 1.0 / (double)nsamp;
        } else {
            for (size_t i = 0; i < nlags; i++) {
                int lag = (int)i - maxlag;
                int mx = lag > kmax ? lag : kmax;
                int mn = lag < kmin ? lag : kmin;
                scale[i] = 1.0 / (double)((long)nsamp - mx + mn);
            }
        }

        for (size_t k = 0; k < nrecs; k++) {
            size_t base = k * nadv;
            if (base + nsamp > N) {
                free(y_cum);
                free(tmp);
                free(scale);
                free(ws);
                free(xs);
                free(ys);
                free(zs);
                free(cys_buf);
                free(ziv);
                free(R_wx);
                free(R_zx);
                free(M_yx);
                return SP_ERR_SHAPE;
            }
            extract_seg(ws, w, N, cols, base, nsamp);
            extract_seg(xs, x, N, cols, base, nsamp);
            extract_seg(ys, y, N, cols, base, nsamp);
            extract_seg(zs, z, N, cols, base, nsamp);
            mean_remove(ws, nsamp);
            mean_remove(xs, nsamp);
            mean_remove(ys, nsamp);
            mean_remove(zs, nsamp);
            for (size_t n = 0; n < nsamp; n++) cys_buf[n] = conj(ys[n]);

            for (size_t i = 0; i < nlags; i++) tmp[i] = 0.0;
            sp_c128 R_zy = 0.0, R_wy = 0.0, M_wz = 0.0;

            for (size_t n = 0; n < nsamp; n++) ziv[n] = 0.0;
            /* ziv(n) = w(n) y*(n+k1) z(n+k2) */
            if (k1 >= 0) {
                for (size_t n = 0; n + (size_t)k1 < nsamp; n++)
                    ziv[n] = ws[n] * cys_buf[n + (size_t)k1];
                for (size_t n = 0; n + (size_t)k1 < nsamp; n++)
                    R_wy += conj(ws[n]) * ys[n + (size_t)k1];
            } else {
                for (size_t n = (size_t)(-k1); n < nsamp; n++)
                    ziv[n] = ws[n] * cys_buf[n + (size_t)k1];
                for (size_t n = (size_t)(-k1); n < nsamp; n++)
                    R_wy += conj(ws[n]) * ys[n + (size_t)k1];
            }
            if (k2 >= 0) {
                for (size_t n = 0; n + (size_t)k2 < nsamp; n++)
                    ziv[n] *= zs[n + (size_t)k2];
                for (size_t n = (size_t)nsamp - (size_t)k2; n < nsamp; n++)
                    ziv[n] = 0.0;
                for (size_t n = 0; n + (size_t)k2 < nsamp; n++)
                    M_wz += conj(ws[n]) * zs[n + (size_t)k2];
            } else {
                for (size_t n = (size_t)(-k2); n < nsamp; n++)
                    ziv[n] *= zs[n + (size_t)k2];
                for (size_t n = 0; n < (size_t)(-k2); n++) ziv[n] = 0.0;
                for (size_t n = (size_t)(-k2); n < nsamp; n++)
                    M_wz += conj(ws[n]) * zs[n + (size_t)k2];
            }
            if (k1 - k2 >= 0) {
                for (size_t n = 0; n + (size_t)(k1 - k2) < nsamp; n++)
                    R_zy += conj(zs[n]) * ys[n + (size_t)(k1 - k2)];
            } else {
                for (size_t n = (size_t)(k2 - k1); n < nsamp; n++)
                    R_zy += conj(zs[n]) * ys[n - (size_t)(k2 - k1)];
            }

            sp_c128 acc = 0.0;
            for (size_t n = 0; n < nsamp; n++) acc += conj(ziv[n]) * xs[n];
            tmp[zlag] += acc;
            for (int m = 1; m <= maxlag; m++) {
                sp_c128 a = 0.0, b = 0.0;
                for (size_t n = (size_t)m; n < nsamp; n++)
                    a += conj(ziv[n]) * xs[n - (size_t)m];
                for (size_t n = 0; n + (size_t)m < nsamp; n++)
                    b += conj(ziv[n]) * xs[n + (size_t)m];
                tmp[zlag - m] += a;
                tmp[zlag + m] += b;
            }
            for (size_t i = 0; i < nlags; i++) y_cum[i] += tmp[i] * scale[i];

            sp_status st;
            st = sp_cum2x(ws, xs, nsamp, 1, maxlag, nsamp, 0, biased, R_wx);
            if (st == SP_OK)
                st = sp_cum2x(zs, xs, nsamp, 1, maxlag + abs(k2), nsamp, 0,
                              biased, R_zx);
            if (st == SP_OK)
                st = sp_cum2x(cys_buf, xs, nsamp, 1, maxlag + abs(k1), nsamp,
                              0, biased, M_yx);
            if (st != SP_OK) {
                free(y_cum);
                free(tmp);
                free(scale);
                free(ws);
                free(xs);
                free(ys);
                free(zs);
                free(cys_buf);
                free(ziv);
                free(R_wx);
                free(R_zx);
                free(M_yx);
                return st;
            }

            for (int m = -maxlag; m <= maxlag; m++) {
                sp_c128 t1 = R_zy * R_wx[m + maxlag] * sc12;
                sp_c128 t2 = R_wy * R_zx[m - k2 + maxlag + abs(k2)] * sc1;
                sp_c128 t3 = M_wz * M_yx[m - k1 + maxlag + abs(k1)] * sc2;
                y_cum[m + maxlag] -= t1 + t2 + t3;
            }
        }
        free(ws);
        free(xs);
        free(ys);
        free(zs);
        free(cys_buf);
        free(ziv);
        free(R_wx);
        free(R_zx);
        free(M_yx);
        free(tmp);
        free(scale);

        for (size_t i = 0; i < nlags; i++) y_cum[i] /= (double)nrecs;
        memcpy(out, y_cum, nlags * sizeof(sp_c128));
        free(y_cum);
        return SP_OK;
    }
}
