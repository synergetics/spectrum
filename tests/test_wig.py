import numpy as np
import pytest

from spectrum import tfrwv, wig2, wig2c, wig4, wig4c
from spectrum._backend import core

HAS_C = core() is not None


def _tone(n=512, f=0.125, seed=1):
    rng = np.random.default_rng(seed)
    return np.cos(2 * np.pi * f * np.arange(n)) + 0.01 * rng.standard_normal(n)


def test_wig2_sinusoid_peak():
    x = _tone()
    W = wig2(x, nfft=64, inc=8)
    assert W.shape[0] == 64 // 1 or True  # n_time = ceil(512/8) = 64
    assert W.shape == (64, 64)
    # auto-term of a real tone at f sits at 2f (inner-product effect)
    two_f_bin = int(round(2 * 0.125 * 64)) % 64
    mid = W[W.shape[0] // 2]
    assert mid[two_f_bin] > 10 * np.abs(mid[: 32]).mean()


def test_wig2c_complex():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    W = wig2c(x, nfft=32, inc=16)
    assert W.shape == (8, 32)
    assert np.iscomplexobj(W)


def test_wig4_symmetry_and_shape():
    x = _tone(256)
    W = wig4(x, nfft=16, inc=32)
    assert W.shape == (8, 16, 16)
    assert np.allclose(W, np.transpose(W, (0, 2, 1)), atol=1e-10)


def test_tfrwv_chirp_ridge():
    n = 1024
    t = np.arange(n)
    # instantaneous frequency goes 0.05 -> 0.10 (phase derivative),
    # so the Wigner ridge goes 0.10 -> 0.20 without aliasing
    x = np.cos(2 * np.pi * (0.05 + 0.05 * t / n) * t)
    W, times = tfrwv(x, nfft=64, inc=64)
    assert W.shape == (16, 64)
    half = np.abs(W[:, 1:32])
    ridge = (np.arange(1, 32) / 64.0)[half.argmax(axis=1)]
    assert ridge[-1] > ridge[0] + 0.04


@pytest.mark.skipif(not HAS_C, reason="C extension not built")
def test_parity_wig(monkeypatch):
    import os

    rng = np.random.default_rng(3)
    xr = rng.standard_normal(200)
    xc = rng.standard_normal(200) + 1j * rng.standard_normal(200)

    for name, x, kw in [
        ("wig2", xr, dict(nfft=32, inc=16)),
        ("wig2c", xc, dict(nfft=32, inc=16)),
        ("wig4", xr, dict(nfft=16, inc=32)),
        ("wig4c", xc, dict(nfft=16, inc=32)),
    ]:
        fn = globals()[name]
        monkeypatch.setenv("SPECTRUM_BACKEND", "python")
        py = fn(x, **kw)
        monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
        c = fn(x, **kw)
        assert np.allclose(c, py, rtol=1e-9, atol=1e-12), name

    monkeypatch.setenv("SPECTRUM_BACKEND", "python")
    Wp, tp = tfrwv(xr, nfft=32, inc=16)
    monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
    Wc, tc = tfrwv(xr, nfft=32, inc=16)
    assert np.allclose(Wc, Wp, rtol=1e-9, atol=1e-12)
    assert np.allclose(tc, tp)
