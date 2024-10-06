# Higher-Order Spectral Analysis Package

"""
Cumulant Estimation Package

This package provides functions for estimating various orders of cumulants
and related measures for time series analysis and signal processing.

Main functions:
---------------
bicoherence : Estimate bicoherence using the direct (FFT) method.
bicoherencex : Estimate cross-bicoherence using the direct (FFT) method.
bispectrumd : Estimate the bispectrum using the direct (FFT) method.
bispectrumdx : Estimate the cross-bispectrum using the direct (FFT) method.
bispectrumi : Estimate the bispectrum using the indirect (time-domain) method.
cum2est : Estimate the covariance (2nd order cumulant) function.
cum2x : Estimate the cross-covariance (2nd order cross-cumulant) function.
cum3est : Estimate the third-order cumulants of a signal.
cum3x : Estimate the third-order cross-cumulants of three signals.
cum4est : Estimate the fourth-order cumulants of a signal.
cum4x : Estimate the fourth-order cross-cumulants of four signals.
cumest : Unified interface for estimating 2nd, 3rd, or 4th order cumulants.

Each function provides detailed documentation on its usage and parameters.
"""

from .bicoherence import bicoherence, plot_bicoherence
from .bicoherencex import bicoherencex, plot_bicoherencex
from .bispectrumd import bispectrumd, plot_bispectrumd
from .bispectrumdx import bispectrumdx, plot_bispectrumdx
from .bispectrumi import bispectrumi, plot_bispectrumi
from .cum2est import cum2est
from .cum2x import cum2x
from .cum3est import cum3est
from .cum3x import cum3x
from .cum4est import cum4est
from .cum4x import cum4x
from .cumest import cumest
from spectrum.matlab import nextpow2, flat_eq, make_arr, shape

__all__ = [
    "bicoherence",
    "bicoherencex",
    "bispectrumd",
    "bispectrumdx",
    "bispectrumi",
    "plot_bicoherence",
    "plot_bicoherencex",
    "plot_bispectrumd",
    "plot_bispectrumdx",
    "plot_bispectrumi",
    "cum2est",
    "cum2x",
    "cum3est",
    "cum3x",
    "cum4est",
    "cum4x",
    "cumest",
    "nextpow2",
    "flat_eq",
    "make_arr",
    "shape",
]
