/* estimators_ext.c - extension kernels (doa cumulant matrix, and later
 * armaqs/armarts support). Inputs are C-contiguous complex128. */
#include "estimators_ext.h"

#include <stdlib.h>
#include <string.h>

sp_status sp_doa_cum(const sp_c128 *X, size_t N, size_t M, sp_c128 *C) {
    /* C[i, j] = cum(x_i, x_j*, x_0, x_0*) with reference sensor 0,
     * exactly mirroring spectrum.doa._cumulant_matrix */
    if (!X || !C || N == 0 || M == 0) return SP_ERR_ARG;
    sp_c128 *t1 = (sp_c128 *)calloc(M * M, sizeof(sp_c128));
    sp_c128 *m1 = (sp_c128 *)calloc(M * M, sizeof(sp_c128));
    sp_c128 *v = (sp_c128 *)calloc(M, sizeof(sp_c128));
    sp_c128 *u = (sp_c128 *)calloc(M, sizeof(sp_c128));
    double mrr = 0.0;
    if (!t1 || !m1 || !v || !u) {
        free(t1);
        free(m1);
        free(v);
        free(u);
        return SP_ERR_MEM;
    }
    for (size_t n = 0; n < N; n++) {
        const sp_c128 *row = X + n * M;
        double wr = creal(row[0]) * creal(row[0]) + cimag(row[0]) * cimag(row[0]);
        mrr += wr;
        for (size_t i = 0; i < M; i++) {
            sp_c128 xi = row[i];
            v[i] += xi * row[0];
            u[i] += conj(xi) * row[0];
            for (size_t j = 0; j < M; j++) {
                t1[i * M + j] += xi * conj(row[j]) * wr;
                m1[i * M + j] += xi * conj(row[j]);
            }
        }
    }
    mrr /= (double)N;
    for (size_t i = 0; i < M; i++) {
        v[i] /= (double)N;
        u[i] /= (double)N;
        for (size_t j = 0; j < M; j++) {
            t1[i * M + j] /= (double)N;
            m1[i * M + j] /= (double)N;
        }
    }
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < M; j++) {
            C[i * M + j] = t1[i * M + j] - m1[i * M + j] * mrr
                           - v[i] * conj(v[j]) - conj(u[i]) * u[j];
        }
    free(t1);
    free(m1);
    free(v);
    free(u);
    return SP_OK;
}
