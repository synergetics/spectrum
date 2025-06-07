#!/usr/bin/env python

import numpy as np
import logging
from typing import Any, Optional, Union, List


log = logging.getLogger(__file__)


def nlgen(
    N: int,
    model_type: str = "bilinear",
    a: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]] = [0.5],
    b: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]] = [1.0],
    c: Union[float, List[float], np.ndarray[Any, np.dtype[Any]]] = [0.1],
    sigma: float = 1.0,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Generate nonlinear time series models.

    Generates various types of nonlinear time series including bilinear models,
    threshold autoregressive models, and other nonlinear structures commonly
    used in testing higher-order spectral analysis methods.

    Parameters:
    -----------
    N : int
        Number of samples to generate.
    model_type : str, optional
        Type of nonlinear model:
        - 'bilinear': Bilinear model
        - 'tar': Threshold autoregressive model
        - 'volterra': Volterra series model
        - 'narma': Nonlinear ARMA model
        - 'henon': Hénon map
        - 'logistic': Logistic map
        Default is 'bilinear'.
    a : float, list, or np.ndarray, optional
        Linear AR coefficients. Default is [0.5].
    b : float, list, or np.ndarray, optional
        MA coefficients or other model parameters. Default is [1.0].
    c : float, list, or np.ndarray, optional
        Nonlinear coefficients. Default is [0.1].
    sigma : float, optional
        Standard deviation of driving noise. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    **kwargs : optional
        Additional model-specific parameters.

    Returns:
    --------
    y : np.ndarray[Any, np.dtype[Any]]
        Generated nonlinear time series of length N.

    Notes:
    ------
    Bilinear model:
    y(n) = sum_i a_i * y(n-i) + sum_j b_j * e(n-j) + sum_i,j c_ij * y(n-i) * e(n-j) + e(n)

    Threshold AR model:
    y(n) = a1 * y(n-1) + e(n)  if y(n-d) <= threshold
    y(n) = a2 * y(n-1) + e(n)  if y(n-d) > threshold

    where e(n) is white Gaussian noise.
    """

    if seed is not None:
        np.random.seed(seed)

    # Convert coefficients to arrays
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    c = np.atleast_1d(c)

    # Generate driving noise
    e = np.random.normal(0, sigma, N + max(len(a), len(b)) + 10)

    # Initialize output
    y = np.zeros(N + max(len(a), len(b)) + 10)

    if model_type.lower() == "bilinear":
        y = _generate_bilinear(y, e, a, b, c, N)
    elif model_type.lower() == "tar":
        y = _generate_tar(y, e, a, c, N, **kwargs)
    elif model_type.lower() == "volterra":
        y = _generate_volterra(y, e, a, b, c, N, **kwargs)
    elif model_type.lower() == "narma":
        y = _generate_narma(y, e, a, b, c, N, **kwargs)
    elif model_type.lower() == "henon":
        y = _generate_henon(N, a, b, sigma, **kwargs)
    elif model_type.lower() == "logistic":
        y = _generate_logistic(N, a, sigma, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return y[-N:]


def _generate_bilinear(y: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, N: int) -> np.ndarray:
    """Generate bilinear model."""

    p = len(a)  # AR order
    q = len(b)  # MA order
    start_idx = max(p, q) + 5

    for n in range(start_idx, len(y)):
        # Linear AR part
        ar_sum = 0
        for i in range(p):
            if n - i - 1 >= 0:
                ar_sum += a[i] * y[n - i - 1]

        # MA part
        ma_sum = 0
        for j in range(q):
            if n - j >= 0:
                ma_sum += b[j] * e[n - j]

        # Bilinear part (simplified: only first nonlinear coefficient)
        bilinear_sum = 0
        if len(c) > 0 and n >= 1:
            bilinear_sum = c[0] * y[n - 1] * e[n]

        y[n] = ar_sum + ma_sum + bilinear_sum + e[n]

    return y


def _generate_tar(y: np.ndarray, e: np.ndarray, a: np.ndarray, c: np.ndarray, N: int, **kwargs) -> np.ndarray:
    """Generate threshold autoregressive model."""

    threshold = kwargs.get("threshold", 0.0)
    delay = kwargs.get("delay", 1)
    a2 = kwargs.get("a2", a * 1.5)  # Different coefficients for upper regime

    a2 = np.atleast_1d(a2)
    p = len(a)
    start_idx = max(p, delay) + 5

    for n in range(start_idx, len(y)):
        # Determine regime based on threshold
        if n - delay >= 0 and y[n - delay] <= threshold:
            # Lower regime
            coeffs = a
        else:
            # Upper regime
            coeffs = a2

        # Apply AR model with selected coefficients
        ar_sum = 0
        for i in range(len(coeffs)):
            if n - i - 1 >= 0:
                ar_sum += coeffs[i] * y[n - i - 1]

        y[n] = ar_sum + e[n]

    return y


def _generate_volterra(
    y: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, N: int, **kwargs
) -> np.ndarray:
    """Generate Volterra series model."""

    order = kwargs.get("order", 2)
    memory = kwargs.get("memory", 3)

    start_idx = memory + 5

    for n in range(start_idx, len(y)):
        output = 0

        # Linear part (first-order Volterra)
        for i in range(min(len(a), memory)):
            if n - i >= 0:
                output += a[i] * e[n - i]

        # Quadratic part (second-order Volterra)
        if order >= 2 and len(c) > 0:
            quad_sum = 0
            idx = 0
            for i in range(memory):
                for j in range(i, memory):
                    if idx < len(c) and n - i >= 0 and n - j >= 0:
                        quad_sum += c[idx] * e[n - i] * e[n - j]
                    idx += 1
            output += quad_sum

        y[n] = output

    return y


def _generate_narma(
    y: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, N: int, **kwargs
) -> np.ndarray:
    """Generate nonlinear ARMA model."""

    nonlinear_func = kwargs.get("nonlinear_func", "tanh")

    p = len(a)
    q = len(b)
    start_idx = max(p, q) + 5

    for n in range(start_idx, len(y)):
        # Linear AR part
        ar_sum = 0
        for i in range(p):
            if n - i - 1 >= 0:
                ar_sum += a[i] * y[n - i - 1]

        # Linear MA part
        ma_sum = 0
        for j in range(q):
            if n - j >= 0:
                ma_sum += b[j] * e[n - j]

        # Nonlinear transformation
        linear_sum = ar_sum + ma_sum

        if nonlinear_func == "tanh":
            nonlinear_part = c[0] * np.tanh(linear_sum) if len(c) > 0 else 0
        elif nonlinear_func == "sigmoid":
            nonlinear_part = c[0] / (1 + np.exp(-np.clip(linear_sum, -500, 500))) if len(c) > 0 else 0
        elif nonlinear_func == "cubic":
            # Clip to prevent overflow
            clipped_sum = np.clip(linear_sum, -10, 10)
            nonlinear_part = c[0] * clipped_sum**3 if len(c) > 0 else 0
        else:
            # Quadratic case
            clipped_sum = np.clip(linear_sum, -100, 100)
            nonlinear_part = c[0] * clipped_sum**2 if len(c) > 0 else 0

        result = linear_sum + nonlinear_part + e[n]
        # Clip final result to prevent divergence
        y[n] = np.clip(result, -1e6, 1e6)

    return y


def _generate_henon(N: int, a: np.ndarray, b: np.ndarray, sigma: float, **kwargs) -> np.ndarray:
    """Generate Hénon map."""

    # Hénon map parameters
    a_param = a[0] if len(a) > 0 else 1.4
    b_param = b[0] if len(b) > 0 else 0.3

    y = np.zeros(N + 2)
    y[0] = kwargs.get("x0", 0.1)
    y[1] = kwargs.get("y0", 0.1)

    for n in range(2, N + 2):
        noise = np.random.normal(0, sigma) if sigma > 0 else 0
        y[n] = 1 - a_param * y[n - 1] ** 2 + b_param * y[n - 2] + noise

    return y[2:]


def _generate_logistic(N: int, a: np.ndarray, sigma: float, **kwargs) -> np.ndarray:
    """Generate logistic map."""

    # Logistic map parameter
    r = a[0] if len(a) > 0 else 3.8

    y = np.zeros(N + 1)
    y[0] = kwargs.get("x0", 0.5)

    for n in range(1, N + 1):
        noise = np.random.normal(0, sigma) if sigma > 0 else 0
        y[n] = r * y[n - 1] * (1 - y[n - 1]) + noise
        # Keep values in [0,1] range
        y[n] = np.clip(y[n], 0, 1)

    return y[1:]


def plot_nonlinear_series(y: np.ndarray[Any, np.dtype[Any]], model_type: str = "Nonlinear", fs: float = 1.0) -> None:
    """
    Plot nonlinear time series with phase space reconstruction.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Time series to plot.
    model_type : str, optional
        Model type for plot title.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    """
    try:
        import matplotlib.pyplot as plt

        N = len(y)
        t = np.arange(N) / fs

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Time series plot
        ax1.plot(t, y, "b-", alpha=0.7)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"{model_type} Time Series")
        ax1.grid(True, alpha=0.3)

        # Phase space reconstruction (embedding dimension 2)
        if N > 1:
            ax2.plot(y[:-1], y[1:], "b.", alpha=0.5, markersize=2)
            ax2.set_xlabel("y(n)")
            ax2.set_ylabel("y(n+1)")
            ax2.set_title("Phase Space (lag=1)")
            ax2.grid(True, alpha=0.3)

        # Histogram
        ax3.hist(y, bins=50, alpha=0.7, density=True)
        ax3.set_xlabel("Amplitude")
        ax3.set_ylabel("Density")
        ax3.set_title("Amplitude Distribution")
        ax3.grid(True, alpha=0.3)

        # Power spectrum
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(N, 1 / fs)
        pos_mask = freqs >= 0
        ax4.semilogy(freqs[pos_mask], np.abs(Y[pos_mask]) ** 2, "b-", alpha=0.7)
        ax4.set_xlabel("Frequency")
        ax4.set_ylabel("Power")
        ax4.set_title("Power Spectrum")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        log.warning("Matplotlib not available for plotting")


def nonlinear_measures(y: np.ndarray[Any, np.dtype[Any]]) -> dict:
    """
    Compute nonlinearity measures for time series.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input time series.

    Returns:
    --------
    measures : dict
        Dictionary containing various nonlinearity measures.
    """

    measures = {}

    # Basic statistics
    measures["mean"] = np.mean(y)
    measures["std"] = np.std(y)
    measures["skewness"] = _skewness(y)
    measures["kurtosis"] = _kurtosis(y)

    # Nonlinearity tests
    measures["bds_statistic"] = _bds_test(y)
    measures["lyapunov_exponent"] = _lyapunov_estimate(y)

    return measures


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    x_centered = x - np.mean(x)
    return np.mean(x_centered**3) / (np.std(x) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    x_centered = x - np.mean(x)
    return np.mean(x_centered**4) / (np.std(x) ** 4) - 3


def _bds_test(x: np.ndarray, m: int = 2, eps: Optional[float] = None) -> float:
    """Simplified BDS test statistic."""
    if eps is None:
        eps = np.std(x) * 0.5

    N = len(x)
    if N < m + 1:
        return 0.0

    # Embed the time series
    embedded = np.array([x[i : i + m] for i in range(N - m + 1)])

    # Compute correlation sum
    C_m = 0.0
    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            if np.max(np.abs(embedded[i] - embedded[j])) < eps:
                C_m += 1

    C_m = C_m / (len(embedded) * (len(embedded) - 1) // 2)

    # This is a simplified version - full BDS test requires more computation
    return C_m


def _lyapunov_estimate(x: np.ndarray, tau: int = 1, m: int = 3) -> float:
    """Estimate largest Lyapunov exponent using Rosenstein's method."""
    N = len(x)
    if N < m + tau * (m - 1) + 1:
        return 0.0

    # Phase space reconstruction
    embedded = np.array([x[i : i + tau * m : tau] for i in range(N - tau * (m - 1))])

    # Find nearest neighbors
    divergence = []

    for i in range(len(embedded) - 1):
        distances = np.array([np.linalg.norm(embedded[i] - embedded[j]) for j in range(len(embedded)) if j != i])

        if len(distances) > 0:
            nearest_idx = np.argmin(distances)
            if nearest_idx >= i:
                nearest_idx += 1

            # Track divergence
            for k in range(1, min(10, len(embedded) - int(max(i, nearest_idx)))):
                if i + k < len(embedded) and nearest_idx + k < len(embedded):
                    dist = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                    if dist > 0:
                        divergence.append(np.log(dist))

    # Estimate Lyapunov exponent from linear fit
    if len(divergence) > 1:
        return np.mean(np.diff(divergence))
    else:
        return 0.0
