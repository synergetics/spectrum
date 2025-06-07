#!/usr/bin/env python

import numpy as np
import logging
from typing import Any, Tuple, Optional

from .tools.matlab import shape
from .armafit import armafit

log = logging.getLogger(__file__)


def armasel(
    y: np.ndarray[Any, np.dtype[Any]],
    pmax: int,
    qmax: int,
    criterion: str = "aic",
    maxlag: Optional[int] = None,
    nsamp: Optional[int] = None,
    overlap: int = 0,
    flag: str = "biased",
) -> Tuple[int, int, float, np.ndarray[Any, np.dtype[Any]]]:
    """
    ARMA model order selection using information criteria.

    Selects the optimal ARMA(p,q) model order by fitting models with
    different orders and comparing them using information criteria.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector (column).
    pmax : int
        Maximum AR order to test.
    qmax : int
        Maximum MA order to test.
    criterion : str, optional
        Information criterion to use:
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
        - 'hq': Hannan-Quinn Information Criterion
        Default is 'aic'.
    maxlag : int, optional
        Maximum lag for cumulant computation. Default is max(pmax,qmax)+10.
    nsamp : int, optional
        Samples per segment for cumulant estimation. Default is len(y).
    overlap : int, optional
        Percentage overlap of segments. Default is 0.
    flag : str, optional
        'biased' or 'unbiased' estimates. Default is 'biased'.

    Returns:
    --------
    p_opt : int
        Optimal AR order.
    q_opt : int
        Optimal MA order.
    ic_min : float
        Minimum information criterion value.
    ic_matrix : np.ndarray[Any, np.dtype[Any]]
        Matrix of information criterion values for all tested orders.
        ic_matrix[i,j] corresponds to ARMA(i,j).

    Notes:
    ------
    The function tests all combinations of AR orders from 0 to pmax and
    MA orders from 0 to qmax. For each combination, it fits an ARMA model
    and computes the specified information criterion.

    Information criteria:
    - AIC = log(σ²) + 2k/N
    - BIC = log(σ²) + k*log(N)/N
    - HQ = log(σ²) + 2k*log(log(N))/N

    where σ² is the residual variance, k is the number of parameters,
    and N is the number of observations.
    """

    (n1, n2) = shape(y, 2)
    N = n1 * n2

    if maxlag is None:
        maxlag = max(pmax, qmax) + 10
    if nsamp is None:
        nsamp = N

    # Initialize results matrix
    ic_matrix = np.full((pmax + 1, qmax + 1), np.inf)

    # Test all combinations of p and q
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            try:
                # Skip (0,0) case
                if p == 0 and q == 0:
                    continue

                # Fit ARMA model
                a, b, rho = armafit(y, p, q, maxlag, nsamp, overlap, flag)

                # Number of parameters
                k = p + q

                # Compute information criterion
                if rho > 0:
                    log_likelihood = -0.5 * N * (np.log(2 * np.pi) + np.log(rho) + 1)

                    if criterion.lower() == "aic":
                        ic = -2 * log_likelihood / N + 2 * k / N
                    elif criterion.lower() == "bic":
                        ic = -2 * log_likelihood / N + k * np.log(N) / N
                    elif criterion.lower() == "hq":
                        ic = -2 * log_likelihood / N + 2 * k * np.log(np.log(N)) / N
                    else:
                        raise ValueError(f"Unknown criterion: {criterion}")

                    ic_matrix[p, q] = ic

            except Exception as e:
                log.warning(f"Failed to fit ARMA({p},{q}): {e}")
                continue

    # Find optimal order
    min_idx = np.unravel_index(np.argmin(ic_matrix), ic_matrix.shape)
    p_opt = min_idx[0]
    q_opt = min_idx[1]
    ic_min = ic_matrix[p_opt, q_opt]

    return int(p_opt), int(q_opt), ic_min, ic_matrix


def plot_ic_surface(
    ic_matrix: np.ndarray[Any, np.dtype[Any]], criterion: str = "AIC", title: Optional[str] = None
) -> None:
    """
    Plot information criterion surface for ARMA order selection.

    Parameters:
    -----------
    ic_matrix : np.ndarray[Any, np.dtype[Any]]
        Matrix of information criterion values.
    criterion : str, optional
        Name of the criterion for labeling. Default is 'AIC'.
    title : str, optional
        Plot title. If None, a default title is generated.
    """
    try:
        import matplotlib.pyplot as plt

        if title is None:
            title = f"{criterion} Surface for ARMA Order Selection"

        pmax, qmax = ic_matrix.shape
        pmax -= 1
        qmax -= 1

        # Create meshgrid
        P, Q = np.meshgrid(range(pmax + 1), range(qmax + 1), indexing="ij")

        # Mask infinite values
        Z = ic_matrix.copy()
        Z[np.isinf(Z)] = np.nan

        # 3D surface plot
        fig = plt.figure(figsize=(12, 5))

        # 3D surface
        ax1 = fig.add_subplot(121, projection="3d")
        surf = ax1.plot_surface(P, Q, Z, cmap="viridis", alpha=0.8)
        ax1.set_xlabel("AR Order (p)")
        ax1.set_ylabel("MA Order (q)")
        ax1.set_zlabel(f"{criterion} Value")
        ax1.set_title(f"3D {title}")

        # 2D contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(P, Q, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)

        # Mark minimum
        min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax2.plot(min_idx[0], min_idx[1], "ro", markersize=10, label=f"Minimum at ({min_idx[0]},{min_idx[1]})")

        ax2.set_xlabel("AR Order (p)")
        ax2.set_ylabel("MA Order (q)")
        ax2.set_title(f"Contour {title}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        log.warning("Matplotlib not available for plotting")


def plot_ic_comparison(
    y: np.ndarray[Any, np.dtype[Any]], pmax: int, qmax: int, criteria: list = ["aic", "bic", "hq"], **kwargs
) -> None:
    """
    Compare different information criteria for ARMA order selection.

    Parameters:
    -----------
    y : np.ndarray[Any, np.dtype[Any]]
        Input data vector.
    pmax : int
        Maximum AR order to test.
    qmax : int
        Maximum MA order to test.
    criteria : list, optional
        List of criteria to compare. Default is ['aic', 'bic', 'hq'].
    **kwargs : optional
        Additional arguments passed to armasel.
    """
    try:
        import matplotlib.pyplot as plt

        results = {}
        for criterion in criteria:
            p_opt, q_opt, ic_min, ic_matrix = armasel(y, pmax, qmax, criterion=criterion, **kwargs)
            results[criterion] = {"p_opt": p_opt, "q_opt": q_opt, "ic_min": ic_min, "ic_matrix": ic_matrix}

        fig, axes = plt.subplots(1, len(criteria), figsize=(5 * len(criteria), 4))
        if len(criteria) == 1:
            axes = [axes]

        for i, criterion in enumerate(criteria):
            ic_matrix = results[criterion]["ic_matrix"]
            p_opt = results[criterion]["p_opt"]
            q_opt = results[criterion]["q_opt"]

            # Mask infinite values
            Z = ic_matrix.copy()
            Z[np.isinf(Z)] = np.nan

            P, Q = np.meshgrid(range(pmax + 1), range(qmax + 1), indexing="ij")

            im = axes[i].contourf(P, Q, Z, levels=20, cmap="viridis")
            axes[i].plot(p_opt, q_opt, "ro", markersize=10)
            axes[i].set_xlabel("AR Order (p)")
            axes[i].set_ylabel("MA Order (q)")
            axes[i].set_title(f"{criterion.upper()}: Optimal ({p_opt},{q_opt})")
            axes[i].grid(True, alpha=0.3)

            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.show()

    except ImportError:
        log.warning("Matplotlib not available for plotting")
