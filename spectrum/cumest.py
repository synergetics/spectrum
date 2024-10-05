import numpy as np
import matplotlib.pyplot as plt
from cum2est import cum2est
from cum3est import cum3est
from cum4est import cum4est


def cumest(y, norder=2, maxlag=0, nsamp=None, overlap=0, flag="biased", k1=0, k2=0):
    """
    Unified interface for estimating cumulants of different orders.

    Parameters:
    -----------
    y : array_like
        Input data vector or time-series.
    norder : int, optional
        Cumulant order: 2, 3, or 4. Default is 2.
    maxlag : int, optional
        Maximum lag to be computed. Default is 0.
    nsamp : int, optional
        Samples per segment. If None, nsamp is set to len(y).
    overlap : int, optional
        Percentage overlap of segments (0-99). Default is 0.
    flag : str, optional
        'biased' or 'unbiased'. Default is 'biased'.
    k1, k2 : int, optional
        The fixed lags for 3rd and 4th order cumulants. Default is 0.

    Returns:
    --------
    y_cum : ndarray
        Estimated cumulant:
        - C2(m) for norder=2
        - C3(m,k1) for norder=3
        - C4(m,k1,k2) for norder=4
        where -maxlag <= m <= maxlag
    """
    y = np.asarray(y).squeeze()

    if y.ndim != 1:
        raise ValueError("Input time series must be a 1-D array.")

    if norder not in [2, 3, 4]:
        raise ValueError("Cumulant order must be 2, 3, or 4.")

    if maxlag < 0:
        raise ValueError("maxlag must be non-negative.")

    if nsamp is None or nsamp > len(y):
        nsamp = len(y)

    overlap = np.clip(overlap, 0, 99)

    if norder == 2:
        return cum2est(y, maxlag, nsamp, overlap, flag)
    elif norder == 3:
        return cum3est(y, maxlag, nsamp, overlap, flag, k1)
    else:  # norder == 4
        return cum4est(y, maxlag, nsamp, overlap, flag, k1, k2)


def plot_cumulant(lags, cumulant, norder, k1=0, k2=0, title=None):
    """
    Plot the estimated cumulant.

    Parameters:
    -----------
    lags : array_like
        Lag values.
    cumulant : array_like
        Cumulant values.
    norder : int
        Cumulant order (2, 3, or 4).
    k1, k2 : int, optional
        The fixed lag values for 3rd and 4th order cumulants.
    title : str, optional
        Title for the plot. If None, a default title is used.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cumulant.real, "b-", label="Real")
    if np.iscomplexobj(cumulant):
        plt.plot(lags, cumulant.imag, "r--", label="Imaginary")

    if title is None:
        title = f"{norder}-Order Cumulant"
        if norder == 3:
            title += f" (k1 = {k1})"
        elif norder == 4:
            title += f" (k1 = {k1}, k2 = {k2})"

    plt.title(title)
    plt.xlabel("Lag m")
    plt.ylabel(f"C{norder}(m{',' + str(k1) if norder > 2 else ''}{',' + str(k2) if norder > 3 else ''})")
    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle=":")
    if np.iscomplexobj(cumulant):
        plt.legend()
    plt.show()
