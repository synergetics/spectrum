import scipy.io as sio
import numpy as np
from spectrum import bispectrumdx, plot_bispectrumdx


def test_specific():
    nl1 = sio.loadmat("./tests/demo/nl1.mat")
    dbic = bispectrumdx(nl1["x"], nl1["x"], nl1["y"], 128, 5)


def test_bispectrumdx():
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
    y = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    z = np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))
    x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)  # Reshape to 2D arrays
    Bspec, waxis = bispectrumdx(x, y, z, nfft=256, nsamp=256)
    assert Bspec.shape == (254, 254)
    assert waxis.shape == (256,)
    plot_bispectrumdx(Bspec, waxis[1:-1])
