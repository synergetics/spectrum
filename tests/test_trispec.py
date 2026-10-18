import numpy as np
import pytest

from spectrum import trispec
from spectrum._backend import core

HAS_C = core() is not None


def _qpc_cubic(n=8192, seed=5):
    """Signal with a cubic frequency coupling at 0.1, 0.12 -> 0.22."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    f1, f2 = 0.10, 0.12
    y = (
        np.cos(2 * np.pi * f1 * t + 0.3)
        + np.cos(2 * np.pi * f2 * t + 1.1)
        + 0.5 * np.cos(2 * np.pi * (f1 + f2) * t + 2.0)
        + 0.1 * rng.standard_normal(n)
    )
    return y.reshape(-1, 1)


def test_trispec_shape_and_symmetry():
    y = _qpc_cubic(2048)
    T, w = trispec(y, nfft=64, nsamp=64, overlap=50)
    assert T.shape == (64, 64, 64)
    assert np.allclose(w, np.arange(64) / 64)
    # exact permutation symmetry of the (k1,k2,k3) arguments
    assert np.allclose(T, np.transpose(T, (0, 2, 1)))
    assert np.allclose(T, np.transpose(T, (1, 0, 2)))
    assert np.allclose(T, np.transpose(T, (2, 1, 0)))


def test_trispec_nfft_bump():
    y = _qpc_cubic(4096)
    T, w = trispec(y, nfft=32, nsamp=256)
    assert T.shape[0] == 256  # bumped to >= nsamp


def test_trispec_nfft_guard():
    y = _qpc_cubic(4096)
    with pytest.raises(ValueError):
        trispec(y, nfft=1024, nsamp=2048)


def test_trispec_dc_relation():
    # single segment, known signal: T(0,0,0) = X0^3 * conj(X0)
    y = np.sin(0.37 * np.arange(64)).reshape(-1, 1)
    T, _ = trispec(y, nfft=64, nsamp=64, overlap=0)
    xseg = y.ravel()
    X0 = np.fft.fft(xseg - xseg.mean(), 64)[0] / 64
    assert np.isclose(T[0, 0, 0], X0**3 * np.conj(X0))


@pytest.mark.skipif(not HAS_C, reason="C extension not built")
def test_parity_trispec(monkeypatch):
    import os

    y = _qpc_cubic(2048)
    monkeypatch.setenv("SPECTRUM_BACKEND", "python")
    Tp, wp = trispec(y, nfft=32, nsamp=32, overlap=50)
    monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
    Tc, wc = trispec(y, nfft=32, nsamp=32, overlap=50)
    assert np.allclose(Tc, Tp, rtol=1e-9, atol=1e-12)
    assert np.allclose(wc, wp)
