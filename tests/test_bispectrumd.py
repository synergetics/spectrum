import scipy.io as sio
import numpy as np
from spectrum import bispectrumd, plot_bispectrumd


def test_specific():
    qpc = sio.loadmat("./tests/demo/qpc.mat")
    dbic = bispectrumd(qpc["zmat"], 128, 3, 64, 0)


def test_bispectrumd():
    t = np.linspace(0, 10, 1000)
    f1, f2 = 5, 10
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + 0.5 * np.sin(2 * np.pi * (f1 + f2) * t)
    y += 0.1 * np.random.randn(len(t))
    y = y.reshape(-1, 1)  # Reshape to 2D array
    Bspec, waxis = bispectrumd(y, nfft=256, nsamp=256)
    assert Bspec.shape == (254, 254)
    assert waxis.shape == (256,)
    plot_bispectrumd(Bspec, waxis[1:-1])
