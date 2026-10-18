import numpy as np
import pytest

from spectrum import doa
from spectrum._backend import core

HAS_C = core() is not None


def _two_source_scenario(M=8, N=400, theta1=-20, theta2=30, snr_db=6.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    m = np.arange(M)
    s1 = np.exp(-1j * np.pi * np.sin(np.deg2rad(theta1)) * m)[:, None] * np.exp(1j * (0.1 * t + 0.3))[None, :]
    s2 = np.exp(-1j * np.pi * np.sin(np.deg2rad(theta2)) * m)[:, None] * np.exp(1j * (0.2 * t + 1.1))[None, :]
    noise = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    sigma = 10 ** (-snr_db / 20)
    return (s1 + s2 + sigma * noise).T


def test_doa_two_sources():
    y = _two_source_scenario()
    angles, pseudo = doa(y, nsources=2)
    # find well-separated peaks
    order = np.argsort(pseudo)[::-1]
    peaks = [angles[order[0]]]
    for idx in order[1:]:
        if all(abs(angles[idx] - p) > 10 for p in peaks):
            peaks.append(angles[idx])
        if len(peaks) == 2:
            break
    peaks = np.sort(peaks)
    assert abs(peaks[0] - -20) < 3
    assert abs(peaks[1] - 30) < 3


def test_doa_input_validation():
    y = _two_source_scenario(M=4, N=50)
    with pytest.raises(ValueError):
        doa(y.ravel(), nsources=1)
    with pytest.raises(ValueError):
        doa(y, nsources=4)


@pytest.mark.skipif(not HAS_C, reason="C extension not built")
def test_parity_doa_cumulant(monkeypatch):
    from spectrum.doa import _cumulant_matrix

    y = _two_source_scenario(M=6, N=200, seed=3)
    py = _cumulant_matrix(y)
    c = np.asarray(__import__("spectrum._spectrum_core", fromlist=["x"]).doa_cum(y))
    assert np.allclose(c, py, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not HAS_C, reason="C extension not built")
def test_parity_doa_end_to_end(monkeypatch):
    import os

    y = _two_source_scenario(M=6, N=200, seed=3)
    monkeypatch.setenv("SPECTRUM_BACKEND", "python")
    ang, p_py = doa(y, nsources=2)
    monkeypatch.delenv("SPECTRUM_BACKEND", raising=False)
    ang, p_c = doa(y, nsources=2)
    assert np.allclose(p_c, p_py, rtol=1e-8, atol=1e-10)
