/* fft_iface.h - FFT abstraction over kissfft (or FFTW when built with it).
 *
 * Conventions match numpy:
 *   forward:  X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N)
 *   backward: x[n] = (1/N) * sum_k X[k] * exp(+2*pi*i*k*n/N)   (like np.fft.ifft)
 */
#ifndef SPECTRUM_FFT_IFACE_H
#define SPECTRUM_FFT_IFACE_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Execute in-place on n samples. Thread-safe (internal plan cache is
 * mutex-protected). Returns 0 on success, nonzero on allocation failure. */
int sp_fft_forward(double complex *data, size_t n);
int sp_fft_backward(double complex *data, size_t n);

/* Plan cache housekeeping (mainly for tests / shutdown). */
void sp_fft_clear_cache(void);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRUM_FFT_IFACE_H */
