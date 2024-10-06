# Higher-Order Spectral Analysis Package

"""
Higher-Order Spectral Analysis Package

This package provides functions for estimating various orders of cumulants,
polyspectra, and related measures for time series analysis and signal processing.

Main functions:
---------------
Spectral Analysis:
bicoherence : Estimate bicoherence using the direct (FFT) method.
bicoherencex : Estimate cross-bicoherence using the direct (FFT) method.
bispectrumd : Estimate the bispectrum using the direct (FFT) method.
bispectrumdx : Estimate the cross-bispectrum using the direct (FFT) method.
bispectrumi : Estimate the bispectrum using the indirect (time-domain) method.
trispectrum : Estimate the trispectrum (4th order spectrum).
tricoherence : Estimate the tricoherence (normalized trispectrum).

Cumulant Estimation:
cum2est : Estimate the covariance (2nd order cumulant) function.
cum2x : Estimate the cross-covariance (2nd order cross-cumulant) function.
cum3est : Estimate the third-order cumulants of a signal.
cum3x : Estimate the third-order cross-cumulants of three signals.
cum4est : Estimate the fourth-order cumulants of a signal.
cum4x : Estimate the fourth-order cross-cumulants of four signals.
cumest : Unified interface for estimating 2nd, 3rd, or 4th order cumulants.

ARMA Modeling:
armafit : ARMA parameter estimation via cross-cumulants.
armasel : ARMA model order selection using information criteria.

Signal Generation:
harmgen : Generate harmonics in multiplicative and additive noise.
harmgen_complex : Generate complex harmonics in noise.
nlgen : Generate nonlinear time series (bilinear, TAR, Volterra, etc.).

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
from .armafit import armafit, plot_arma_poles_zeros
from .armasel import armasel, plot_ic_surface, plot_ic_comparison
from .harmgen import harmgen, harmgen_complex, plot_harmonic_signal, harmonic_snr
from .nlgen import nlgen, plot_nonlinear_series, nonlinear_measures
from .trispectrum import (
    trispectrum,
    tricoherence,
    plot_trispectrum,
    plot_tricoherence_summary,
    detect_quadratic_coupling,
)
from .tools.matlab import nextpow2, flat_eq, make_arr, shape

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
    "armafit",
    "armasel",
    "plot_arma_poles_zeros",
    "plot_ic_surface",
    "plot_ic_comparison",
    "harmgen",
    "harmgen_complex",
    "plot_harmonic_signal",
    "harmonic_snr",
    "nlgen",
    "plot_nonlinear_series",
    "nonlinear_measures",
    "trispectrum",
    "tricoherence",
    "plot_trispectrum",
    "plot_tricoherence_summary",
    "detect_quadratic_coupling",
    "nextpow2",
    "flat_eq",
    "make_arr",
    "shape",
]
