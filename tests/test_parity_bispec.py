"""Parity tests: bispectrum/bicoherence/trispectrum C kernels vs the
corrected pure-Python reference estimators."""

import numpy as np
import pytest

from spectrum._backend import core

pytestmark = pytest.mark.skipif(core() is None, reason="C extension not built")

RTOL, ATOL = 1e-9, 1e-12


def compare(monkeypatch, fn):
    monkeypatch.setenv("SPECTRUM_BACKEND", "python")
    py = fn()
    monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
    c = fn()
    for a, b in zip(c, py):
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.shape == b.shape
        assert np.allclose(a, b, rtol=RTOL, atol=ATOL), np.abs(a - b).max()


@pytest.fixture
def data():
    rng = np.random.default_rng(77)
    n = 4096
    return {
        "real": rng.standard_normal(n).reshape(-1, 1),
        "complex": (rng.standard_normal(n) + 1j * rng.standard_normal(n)).reshape(-1, 1),
        "qpc": (
            np.cos(2 * np.pi * 0.10 * np.arange(n))
            + np.cos(2 * np.pi * 0.15 * np.arange(n))
            + 0.5 * np.cos(2 * np.pi * 0.25 * np.arange(n))
            + 0.1 * rng.standard_normal(n)
        ).reshape(-1, 1),
    }


@pytest.mark.parametrize("nsamp,overlap", [(0, 50), (256, 50), (512, 0), (256, 75)])
@pytest.mark.parametrize("nfft", [128, 256])
def test_parity_bicoherence(monkeypatch, data, nfft, nsamp, overlap):
    from spectrum import bicoherence

    y = data["qpc"]
    compare(monkeypatch, lambda: bicoherence(y, nfft=nfft, nsamp=nsamp, overlap=overlap))


@pytest.mark.parametrize("nsamp,overlap", [(0, 50), (256, 50)])
def test_parity_bicoherencex(monkeypatch, data, nsamp, overlap):
    from spectrum import bicoherencex

    w, x, y = data["real"], data["complex"], data["qpc"]
    compare(
        monkeypatch,
        lambda: bicoherencex(w, x, y, nfft=128, nsamp=nsamp, overlap=overlap),
    )


@pytest.mark.parametrize("wind", [3, 5, 9])
@pytest.mark.parametrize("nsamp,overlap", [(256, 50), (0, 50)])
def test_parity_bispectrumd_optimal(monkeypatch, data, wind, nsamp, overlap):
    from spectrum import bispectrumd

    y = data["qpc"]
    compare(
        monkeypatch,
        lambda: bispectrumd(y, nfft=128, wind=wind, nsamp=nsamp, overlap=overlap),
    )


def test_parity_bispectrumd_1d_window(monkeypatch, data):
    from spectrum import bispectrumd

    y = data["qpc"]
    w1 = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    compare(monkeypatch, lambda: bispectrumd(y, nfft=128, wind=w1, nsamp=256, overlap=50))


def test_parity_bispectrumd_2d_window_and_invalid(monkeypatch, data):
    from spectrum import bispectrumd

    y = data["qpc"]
    rng = np.random.default_rng(5)
    w2 = rng.random((5, 5))
    w2 = w2 + w2.T  # square + odd
    compare(monkeypatch, lambda: bispectrumd(y, nfft=128, wind=w2, nsamp=256, overlap=50))
    winv = rng.random((4, 6))
    compare(monkeypatch, lambda: bispectrumd(y, nfft=128, wind=winv, nsamp=256, overlap=50))


def test_parity_bispectrumdx(monkeypatch, data):
    from spectrum import bispectrumdx

    x, y, z = data["real"], data["complex"], data["qpc"]
    compare(monkeypatch, lambda: bispectrumdx(x, y, z, nfft=128, nsamp=256, overlap=50))


@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("wind", [None, 0, 1, -1])
def test_parity_bispectrumi(monkeypatch, data, flag, wind):
    from spectrum import bispectrumi

    y = data["qpc"]
    compare(
        monkeypatch,
        lambda: bispectrumi(y, nlag=24, nsamp=512, overlap=50, flag=flag, nfft=128, wind=wind),
    )


@pytest.mark.parametrize("window", [None, "rect", "hanning", "hamming", "blackman"])
def test_parity_trispectrum(monkeypatch, data, window):
    from spectrum import trispectrum

    y = data["qpc"]
    compare(monkeypatch, lambda: trispectrum(y, nfft=256, window=window, nsamp=512, overlap=0))


@pytest.mark.parametrize("window", [None, "hanning"])
def test_parity_tricoherence(monkeypatch, data, window):
    from spectrum import tricoherence

    y = data["qpc"]
    compare(monkeypatch, lambda: tricoherence(y, nfft=256, window=window, nsamp=512, overlap=0))


def test_parity_complex_bispectrumd(monkeypatch, data):
    from spectrum import bispectrumd

    y = data["complex"]
    compare(monkeypatch, lambda: bispectrumd(y, nfft=128, nsamp=256, overlap=50))
