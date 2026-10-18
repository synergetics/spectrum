/* wigner.h */
#ifndef SPECTRUM_WIGNER_H
#define SPECTRUM_WIGNER_H

#include "estimators_cum.h"

sp_status sp_wig2(const sp_c128 *x, size_t N, size_t nfft, size_t inc,
                  int complex_version, sp_c128 *W);
sp_status sp_wig4(const sp_c128 *x, size_t N, size_t nfft, size_t inc,
                  int complex_version, sp_c128 *W);

#endif /* SPECTRUM_WIGNER_H */
