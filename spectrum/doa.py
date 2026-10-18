"""Direction of arrival estimation via 4th-order cumulant MUSIC.

Implements cumulant-based MUSIC for a uniform linear array with half-
wavelength sensor spacing, following the HOSA toolbox manual and the
Porat-Friedlander / Gonen-Mendel cumulant-MUSIC literature (the
original DOA.M source was not recoverable; the reference-sensor
fourth-order cumulant slice is used):

    C[i, j] = cum(x_i, x_j*, x_ref, x_ref*)

For narrowband non-Gaussian sources this matrix has the Vandermonde
structure C = A diag(gamma_k |a_ref(theta_k)|^2) A^H with rank equal to
the number of sources and is insensitive to Gaussian noise. The
noise subspace is obtained from its eigendecomposition and the
MUSIC pseudospectrum is scanned over angles.
"""

from typing import Any, Tuple

import numpy as np

from ._backend import use_c, ccall


def _cumulant_matrix(y: np.ndarray, ref: int = 0) -> np.ndarray:
    """C[i, j] = cum(x_i, x_j*, x_ref, x_ref*) via its moment expansion:

    cum(w, x*, y, z*) = E[w x* y z*] - E[w x*]E[y z*]
                        - E[w y]E[x* z*] - E[w z*]E[x* y]
    """
    X = np.asarray(y, dtype=complex)
    N, M = X.shape
    r = ref
    xr = X[:, r]
    w_r = np.abs(xr) ** 2

    t1 = (X * w_r[:, None]).T @ X.conj() / N  # E[x_i x_j* |x_r|^2]
    m1 = X.T @ X.conj() / N  # E[x_i x_j*]
    m_rr = float(np.mean(w_r))  # E[x_r x_r*]
    v = X.T @ xr / N  # E[x_i x_r]
    u = X.conj().T @ xr / N  # E[x_j* x_r]

    C = t1 - m1 * m_rr - np.outer(v, np.conj(v)) - np.outer(np.conj(u), u)
    return C


def doa(
    y: np.ndarray[Any, np.dtype[Any]],
    nsources: int = 1,
    nscan: int = 361,
) -> Tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Estimate directions of arrival with 4th-order cumulant MUSIC.

    Parameters:
    -----------
    y : np.ndarray
        Array data with shape (snapshots, sensors). Each column is one
        sensor's time series (complex baseband or real).
    nsources : int
        Number of impinging sources.
    nscan : int
        Number of scan points in [-90, 90] degrees (default 361).

    Returns:
    --------
    angles : np.ndarray
        Scan grid in degrees.
    pseudo : np.ndarray
        MUSIC pseudospectrum over the grid; peaks are the DOAs.
    """
    x = np.asarray(y)
    if x.ndim == 1:
        raise ValueError("y must be 2-D with shape (snapshots, sensors)")
    N, M = x.shape
    if M < 2:
        raise ValueError("need at least 2 sensors")
    if nsources >= M:
        raise ValueError("nsources must be < number of sensors")

    x = np.ascontiguousarray(x, dtype=complex)
    if use_c():
        C = np.asarray(ccall("doa_cum", x))
    else:
        C = _cumulant_matrix(x)

    # Hermitian symmetrization (exact in expectation) + eigendecomposition
    Cs = (C + C.conj().T) / 2
    w, V = np.linalg.eigh(Cs)
    order = np.argsort(np.abs(w))  # source eigenvalues may be negative
    En = V[:, order[: M - nsources]]  # noise subspace

    angles = np.linspace(-90.0, 90.0, nscan)
    m = np.arange(M)
    pseudo = np.empty(nscan)
    for i, th in enumerate(angles):
        a = np.exp(-1j * np.pi * np.sin(np.deg2rad(th)) * m)
        denom = np.sum(np.abs(a.conj() @ En) ** 2)
        pseudo[i] = 1.0 / max(denom, 1e-300)
    return angles, pseudo
