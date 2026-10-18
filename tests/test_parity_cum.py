"""Parity tests: C kernels vs the corrected pure-Python reference.

The pure-Python estimators are the executable reference for "correct HOSA
semantics"; the C extension must reproduce them to rtol=1e-10 across a
parameter grid covering real/complex data, both flags, lag sign
combinations, and both segmentation regimes.
"""

import numpy as np
import pytest

from spectrum._backend import core

pytestmark = pytest.mark.skipif(core() is None, reason="C extension not built")

RTOL, ATOL = 1e-10, 1e-12


def run_python(monkeypatch, fn):
    monkeypatch.setenv("SPECTRUM_BACKEND", "python")
    return fn()


def run_c(monkeypatch, fn):
    monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
    return fn()


def compare(monkeypatch, fn):
    got_c = np.asarray(run_c(monkeypatch, fn))
    got_py = np.asarray(run_python(monkeypatch, fn))
    assert got_c.shape == got_py.shape
    assert np.allclose(got_c, got_py, rtol=RTOL, atol=ATOL), np.abs(got_c - got_py).max()


@pytest.fixture
def data():
    rng = np.random.default_rng(1234)
    return {
        "real": rng.standard_normal(512).reshape(-1, 1),
        "complex": (rng.standard_normal(512) + 1j * rng.standard_normal(512)).reshape(-1, 1),
    }


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("maxlag", [0, 3, 10])
@pytest.mark.parametrize("nsamp,overlap", [(512, 0), (256, 50), (0, 0)])
def test_parity_cum2est(monkeypatch, data, kind, flag, maxlag, nsamp, overlap):
    from spectrum import cum2est

    y = data[kind]
    if nsamp == 0:
        nsamp = 512
    compare(monkeypatch, lambda: cum2est(y, maxlag, nsamp, overlap, flag))


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("maxlag", [0, 5])
@pytest.mark.parametrize("k1", [-3, 0, 2])
def test_parity_cum3est(monkeypatch, data, kind, flag, maxlag, k1):
    from spectrum import cum3est

    y = data[kind]
    compare(monkeypatch, lambda: cum3est(y, maxlag, 512, 0, flag, k1))


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("k1,k2", [(0, 0), (1, 0), (0, 1), (2, -1), (-1, 2), (-2, -2)])
def test_parity_cum4est(monkeypatch, data, kind, flag, k1, k2):
    from spectrum import cum4est

    y = data[kind]
    compare(monkeypatch, lambda: cum4est(y, 4, 512, 0, flag, k1, k2))


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("maxlag,nsamp,overlap", [(0, 512, 0), (3, 256, 50)])
def test_parity_cum2x(monkeypatch, data, kind, flag, maxlag, nsamp, overlap):
    from spectrum import cum2x

    x = data[kind]
    y = data["complex"] if kind == "real" else data["real"]
    compare(monkeypatch, lambda: cum2x(x, y, maxlag, nsamp, overlap, flag))


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("k1", [-2, 0, 3])
def test_parity_cum3x(monkeypatch, data, kind, flag, k1):
    from spectrum import cum3x

    x, y, z = data[kind], data["real"], data["complex"]
    compare(monkeypatch, lambda: cum3x(x, y, z, 3, 512, 0, flag, k1))


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("flag", ["biased", "unbiased"])
@pytest.mark.parametrize("k1,k2", [(0, 0), (1, 1), (2, -1), (-1, 2), (3, 0)])
def test_parity_cum4x(monkeypatch, data, kind, flag, k1, k2):
    from spectrum import cum4x

    w, x, y, z = data[kind], data["real"], data["complex"], data[kind]
    compare(monkeypatch, lambda: cum4x(w, x, y, z, 3, 512, 0, flag, k1, k2))


def test_parity_cumest_dispatch(monkeypatch, data):
    from spectrum import cumest

    for norder in (2, 3, 4):
        y = data["real"]
        compare(monkeypatch, lambda n=norder: cumest(y, n, 4, nsamp=512, k1=1, k2=1))


def test_parity_segmented_grid(monkeypatch, data):
    """Exercise several segment counts and overlaps end-to-end."""
    from spectrum import cum2est, cum3est

    y = data["real"]
    for nsamp in (64, 128, 171):
        for overlap in (0, 25, 50, 75):
            compare(monkeypatch, lambda: cum2est(y, 6, nsamp, overlap, "unbiased"))
            compare(monkeypatch, lambda: cum3est(y, 6, nsamp, overlap, "biased", 1))
