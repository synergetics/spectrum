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
    assert Bspec.shape == (256, 256)
    assert waxis.shape == (256,)
    plot_bispectrumd(Bspec, waxis)


def test_bispectrumd_default_and_windows():
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.standard_normal(4096)
    Bspec, waxis = bispectrumd(y)
    assert Bspec.shape == (Bspec.shape[0], Bspec.shape[0])
    assert len(np.asarray(waxis).ravel()) == Bspec.shape[0]

    # 1-D smoothing window follows MATLAB's zero-padded construction
    w1 = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    Bspec, _ = bispectrumd(y.reshape(-1, 1), nfft=256, nsamp=256, wind=w1)
    assert Bspec.shape == (256, 256)

    # invalid 2-D windows warn and skip smoothing, keeping nfft x nfft
    Bspec, _ = bispectrumd(y.reshape(-1, 1), nfft=256, nsamp=256, wind=rng.random((4, 6)))
    assert Bspec.shape == (256, 256)


def test_bispectrumd_qpc_peak():
    import numpy as np

    rng = np.random.default_rng(1)
    t = np.arange(8192)
    f1, f2 = 0.10, 0.15
    y = (
        np.cos(2 * np.pi * f1 * t)
        + np.cos(2 * np.pi * f2 * t)
        + 0.5 * np.cos(2 * np.pi * (f1 + f2) * t)
        + 0.1 * rng.standard_normal(8192)
    )
    Bspec, waxis = bispectrumd(y.reshape(-1, 1), nfft=256, nsamp=256, overlap=50)
    w = np.asarray(waxis).ravel()
    i = int(np.argmin(np.abs(w - f1)))
    j = int(np.argmin(np.abs(w - f2)))
    assert abs(Bspec[i, j]) > abs(Bspec).max() * 0.5
