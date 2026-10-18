import numpy as np
import pytest

from spectrum import cumtrue
from spectrum._backend import core

HAS_C = core() is not None


def test_cumtrue_identity_ma():
    # ma = [1], ar = [1]: h = [1]
    c2 = cumtrue([1.0], [1.0], norder=2, nlags=2).ravel()
    assert np.allclose(c2, [0, 0, 1, 0, 0])
    c3 = cumtrue([1.0], [1.0], norder=3, nlags=1)
    assert c3[1, 1] == 1.0
    assert c3.shape == (3, 3)


def test_cumtrue_ar1_closed_form():
    # AR(1) a=[1, -0.5]: h(n) = 0.5^n; C2(m) = (4/3) * 0.5^|m|
    c2 = cumtrue([1.0], [1.0, -0.5], norder=2, nlags=3).ravel()
    expect = [(4.0 / 3.0) * 0.5 ** abs(m) for m in range(-3, 4)]
    assert np.allclose(c2, expect, atol=1e-4)  # impulse-response truncation


def test_cumtrue_ma1_hand_computed():
    # MA(1) b=[1, 0.3]: h = [1, 0.3]
    c3 = cumtrue([1.0, 0.3], norder=3, nlags=1)
    # c3(i, j) = sum_n h(n) h(n+i) h(n+j)
    assert np.isclose(c3[1, 1], 1.027)  # c3(0,0) = h0^3 + h1^3
    assert np.isclose(c3[1, 2], 0.3)  # c3(0,1) = h0^2 h1
    assert np.isclose(c3[2, 2], 0.09)  # c3(1,1) = h0 h1^2
    assert np.isclose(c3[1, 0], 0.09)  # c3(0,-1) = h1^2 h0
    assert np.isclose(c3[0, 0], 0.3)  # c3(-1,-1) = h1 h0^2


def test_cumtrue_order4_slice():
    # MA(1) b=[1, 0.3], k=1: c4(i, j, 1) entries
    c4 = cumtrue([1.0, 0.3], norder=4, nlags=1, k=1)
    # c4(0,0,1) = h0^3 h1 = 0.3 -> at (nlags+0, nlags+0) = (1,1)
    assert np.isclose(c4[1, 1], 0.3)


def test_cumtrue_unstable_raises():
    with pytest.raises(ValueError):
        cumtrue([1.0], [1.0, 1.5], norder=2, nlags=2)


def test_cumtrue_matches_empirical():
    # simulate an MA(2) process and compare theoretical vs estimated c3
    rng = np.random.default_rng(42)
    b = [1.0, 0.4, -0.2]
    n = 200000
    # skewed innovations: exponential(1) - 1 has unit third central moment
    e = rng.exponential(1.0, n) - 1.0
    y = e.copy()
    y[1:] += 0.4 * e[:-1]
    y[2:] += -0.2 * e[:-2]
    from spectrum import cumest

    c3_est = cumest(y.reshape(-1, 1), 3, 2, n, 0, "biased").ravel()
    # c3 slice (m, k1=0): compare against cumtrue(norder=3) center column
    col_true = cumtrue(b, norder=3, nlags=2)[:, 2]  # C3(m, 0), m = -2..2
    # exponential(1)-1 innovations have third central moment 2
    assert np.allclose(c3_est / 2.0, col_true, atol=0.05)


@pytest.mark.skipif(not HAS_C, reason="C extension not built")
def test_parity_cumtrue(monkeypatch):
    cases = [
        ([1.0], None, 2, 3, 0),
        ([1.0, 0.3], None, 3, 2, 0),
        ([1.0, 0.4, -0.2], None, 4, 3, 1),
        ([1.0], [1.0, -0.5], 2, 4, 0),
        ([1.0, 0.2], [1.0, -0.3, 0.1], 3, 5, 0),
        ([1.0, 0.2], [1.0, -0.3, 0.1], 4, 4, 2),
    ]
    import os

    for ma, ar, norder, nlags, k in cases:
        monkeypatch.setenv("SPECTRUM_BACKEND", "python")
        py = np.asarray(cumtrue(ma, ar, norder=norder, nlags=nlags, k=k))
        monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
        c = np.asarray(cumtrue(ma, ar, norder=norder, nlags=nlags, k=k))
        assert c.shape == py.shape
        assert np.allclose(c, py, rtol=1e-10, atol=1e-12), (ma, ar, norder)
