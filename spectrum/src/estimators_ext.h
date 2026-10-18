/* estimators_ext.h */
#ifndef SPECTRUM_ESTIMATORS_EXT_H
#define SPECTRUM_ESTIMATORS_EXT_H

#include "estimators_cum.h"

/* C[i, j] = cum(x_i, x_j*, x_0, x_0*); X is (N, M) C-contiguous
 * complex128; C is M x M row-major. */
sp_status sp_doa_cum(const sp_c128 *X, size_t N, size_t M, sp_c128 *C);

#endif /* SPECTRUM_ESTIMATORS_EXT_H */
