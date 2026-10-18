/* test_fft_iface.c - smoke test for the FFT interface (numpy conventions). */
#include "spectrum/src/fft/fft_iface.h"
#include <stdio.h>
#include <complex.h>
#include <math.h>

int main(void) {
    enum { N = 8 };
    double complex x[N], orig[N];
    int fail = 0;

    /* impulse -> flat spectrum of ones */
    for (int i = 0; i < N; i++) x[i] = (i == 0) ? 1.0 : 0.0;
    if (sp_fft_forward(x, N) != 0) { printf("forward failed\n"); return 1; }
    for (int i = 0; i < N; i++)
        if (cabs(x[i] - 1.0) > 1e-12) { printf("bin %d != 1\n", i); fail = 1; }

    /* backward of flat spectrum -> impulse (1/N scaling like np.fft.ifft) */
    if (sp_fft_backward(x, N) != 0) { printf("backward failed\n"); return 1; }
    for (int i = 0; i < N; i++) {
        double expect = (i == 0) ? 1.0 : 0.0;
        if (cabs(x[i] - expect) > 1e-12) { printf("inv %d wrong\n", i); fail = 1; }
    }

    /* shifted impulse: X[k] = e^{-2 pi i k / N}; bin1 = e^{-i pi/4} */
    for (int i = 0; i < N; i++) x[i] = (i == 1) ? 1.0 : 0.0;
    sp_fft_forward(x, N);
    if (cabs(x[1] - (cos(-M_PI / 4.0) + I * sin(-M_PI / 4.0))) > 1e-12) {
        printf("bin1 wrong\n");
        fail = 1;
    }

    /* roundtrip on a random-ish signal */
    for (int i = 0; i < N; i++) {
        orig[i] = sin(0.7 * i) + I * cos(1.3 * i);
        x[i] = orig[i];
    }
    sp_fft_forward(x, N);
    sp_fft_backward(x, N);
    for (int i = 0; i < N; i++)
        if (cabs(x[i] - orig[i]) > 1e-10) { printf("roundtrip %d off\n", i); fail = 1; }

    sp_fft_clear_cache();
    printf(fail ? "FAIL\n" : "OK\n");
    return fail;
}
