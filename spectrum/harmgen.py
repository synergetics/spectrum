#!/usr/bin/env python

import numpy as np
import logging
from typing import Any, Optional, Union, List


log = logging.getLogger(__file__)


def harmgen(
    N: int,
    A: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]],
    f: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]],
    phi: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]] = 0,
    sigma_n: float = 0.0,
    sigma_m: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Generate harmonics in multiplicative and additive noise.

    Generates a signal consisting of multiple sinusoidal components with
    additive and multiplicative noise components. This is useful for
    testing higher-order spectral analysis methods.

    Parameters:
    -----------
    N : int
        Number of samples to generate.
    A : float, list, or np.ndarray
        Amplitude(s) of the harmonic component(s).
    f : float, list, or np.ndarray
        Normalized frequency(ies) of the harmonic component(s) (0 to 0.5).
    phi : float, list, or np.ndarray, optional
        Phase(s) of the harmonic component(s) in radians. Default is 0.
    sigma_n : float, optional
        Standard deviation of additive white Gaussian noise. Default is 0.
    sigma_m : float, optional
        Standard deviation of multiplicative white Gaussian noise. Default is 0.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns:
    --------
    y : np.ndarray[Any, np.dtype[Any]]
        Generated signal of length N.

    Notes:
    ------
    The generated signal has the form:
    y(n) = [sum_k A_k * cos(2*pi*f_k*n + phi_k)] * (1 + sigma_m*w_m(n)) + sigma_n*w_n(n)

    where w_m(n) and w_n(n) are independent white Gaussian noise processes
    with zero mean and unit variance.

    The multiplicative noise component is useful for modeling amplitude
    modulation effects, while the additive noise represents measurement
    noise or other disturbances.

    Examples:
    ---------
    # Single sinusoid with additive noise
    y = harmgen(1000, A=1.0, f=0.1, sigma_n=0.1)

    # Multiple harmonics with different amplitudes and phases
    y = harmgen(1000, A=[1.0, 0.5, 0.3], f=[0.1, 0.2, 0.3],
                phi=[0, np.pi/4, np.pi/2], sigma_n=0.05)
    """

    if seed is not None:
        np.random.seed(seed)

    # Convert inputs to arrays
    A = np.atleast_1d(A)
    f = np.atleast_1d(f)
    phi = np.atleast_1d(phi)

    # Check input dimensions
    n_harmonics = len(A)
    if len(f) != n_harmonics:
        raise ValueError("Length of A and f must be the same")

    if len(phi) == 1 and n_harmonics > 1:
        phi = np.full(n_harmonics, phi[0])
    elif len(phi) != n_harmonics:
        raise ValueError("Length of phi must be 1 or equal to length of A")

    # Check frequency range
    if np.any(f < 0) or np.any(f > 0.5):
        raise ValueError("Normalized frequencies must be between 0 and 0.5")

    # Generate time index
    n = np.arange(N)

    # Generate harmonic signal
    y = np.zeros(N)
    for k in range(n_harmonics):
        y += A[k] * np.cos(2 * np.pi * f[k] * n + phi[k])

    # Add multiplicative noise
    if sigma_m > 0:
        mult_noise = np.random.normal(0, sigma_m, N)
        y = y * (1 + mult_noise)

    # Add additive noise
    if sigma_n > 0:
        add_noise = np.random.normal(0, sigma_n, N)
        y = y + add_noise

    return y


def harmgen_complex(
    N: int,
    A: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]],
    f: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]],
    phi: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]] = 0,
    sigma_n: float = 0.0,
    sigma_m: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Generate complex harmonics in multiplicative and additive noise.

    Similar to harmgen but generates complex exponentials instead of
    real sinusoids.

    Parameters:
    -----------
    N : int
        Number of samples to generate.
    A : float, list, or np.ndarray
        Amplitude(s) of the harmonic component(s).
    f : float, list, or np.ndarray
        Normalized frequency(ies) of the harmonic component(s) (-0.5 to 0.5).
    phi : float, list, or np.ndarray, optional
        Phase(s) of the harmonic component(s) in radians. Default is 0.
    sigma_n : float, optional
        Standard deviation of additive complex white Gaussian noise. Default is 0.
    sigma_m : float, optional
        Standard deviation of multiplicative complex white Gaussian noise. Default is 0.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns:
    --------
    y : np.ndarray[Any, np.dtype[Any]]
        Generated complex signal of length N.

    Notes:
    ------
    The generated signal has the form:
    y(n) = [sum_k A_k * exp(j*(2*pi*f_k*n + phi_k))] * (1 + sigma_m*w_m(n)) + sigma_n*w_n(n)

    where w_m(n) and w_n(n) are independent complex white Gaussian noise processes.
    """

    if seed is not None:
        np.random.seed(seed)

    # Convert inputs to arrays
    A = np.atleast_1d(A)
    f = np.atleast_1d(f)
    phi = np.atleast_1d(phi)

    # Check input dimensions
    n_harmonics = len(A)
    if len(f) != n_harmonics:
        raise ValueError("Length of A and f must be the same")

    if len(phi) == 1 and n_harmonics > 1:
        phi = np.full(n_harmonics, phi[0])
    elif len(phi) != n_harmonics:
        raise ValueError("Length of phi must be 1 or equal to length of A")

    # Check frequency range
    if np.any(f < -0.5) or np.any(f > 0.5):
        raise ValueError("Normalized frequencies must be between -0.5 and 0.5")

    # Generate time index
    n = np.arange(N)

    # Generate complex harmonic signal
    y = np.zeros(N, dtype=complex)
    for k in range(n_harmonics):
        y += A[k] * np.exp(1j * (2 * np.pi * f[k] * n + phi[k]))

    # Add multiplicative noise
    if sigma_m > 0:
        mult_noise_real = np.random.normal(0, sigma_m / np.sqrt(2), N)
        mult_noise_imag = np.random.normal(0, sigma_m / np.sqrt(2), N)
        mult_noise = mult_noise_real + 1j * mult_noise_imag
        y = y * (1 + mult_noise)

    # Add additive noise
    if sigma_n > 0:
        add_noise_real = np.random.normal(0, sigma_n / np.sqrt(2), N)
        add_noise_imag = np.random.normal(0, sigma_n / np.sqrt(2), N)
        add_noise = add_noise_real + 1j * add_noise_imag
        y = y + add_noise

    return y


def plot_harmonic_signal(y: np.ndarray[Any, np.dtype[Any]], fs: float = 1.0, title: str = "Harmonic Signal") -> None:
    """
    Plot time domain and frequency domain representation of harmonic signal.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Signal to plot.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    title : str, optional
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt

        N = len(y)
        t = np.arange(N) / fs

        # Compute FFT
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(N, 1 / fs)

        # Only plot positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        Y_mag = np.abs(Y[pos_mask])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Time domain plot
        if np.iscomplexobj(y):
            ax1.plot(t, np.real(y), "b-", label="Real part", alpha=0.7)
            ax1.plot(t, np.imag(y), "r-", label="Imaginary part", alpha=0.7)
            ax1.plot(t, np.abs(y), "k--", label="Magnitude", alpha=0.7)
            ax1.legend()
        else:
            ax1.plot(t, y, "b-", alpha=0.7)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"{title} - Time Domain")
        ax1.grid(True, alpha=0.3)

        # Frequency domain plot
        ax2.semilogy(freqs, Y_mag, "b-", alpha=0.7)
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Magnitude")
        ax2.set_title(f"{title} - Frequency Domain")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        log.warning("Matplotlib not available for plotting")


def harmonic_snr(
    y: np.ndarray[Any, np.dtype[Any]],
    signal_freqs: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]],
    fs: float = 1.0,
    freq_tol: float = 0.01,
) -> float:
    """
    Estimate signal-to-noise ratio for harmonic signal.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input signal.
    signal_freqs : float, list, or np.ndarray
        Known signal frequencies.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    freq_tol : float, optional
        Frequency tolerance for signal identification. Default is 0.01.

    Returns:
    --------
    snr : float
        Estimated SNR in dB.
    """

    signal_freqs = np.atleast_1d(signal_freqs)

    N = len(y)
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, 1 / fs)
    power_spectrum = np.abs(Y) ** 2

    # Identify signal and noise bins
    signal_power = 0
    noise_power = 0

    for freq in freqs:
        is_signal = False
        for sig_freq in signal_freqs:
            if abs(freq - sig_freq) <= freq_tol or abs(freq + sig_freq) <= freq_tol:
                is_signal = True
                break

        idx = np.where(freqs == freq)[0][0]
        if is_signal:
            signal_power += power_spectrum[idx]
        else:
            noise_power += power_spectrum[idx]

    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = np.inf

    return snr
