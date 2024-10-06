import scipy.io as sio
import numpy as np
from spectrum import cum2x


def test_specific():
    y = sio.loadmat("./tests/demo/ma1.mat")["y"]

    biased = cum2x(y, y, 3, 100, 0, "biased").flatten().tolist()
    unbiased = cum2x(y, y, 3, 100, 0, "unbiased").flatten().tolist()

    assert biased == [
        -0.25719315012231203,
        -0.12011232042321396,
        0.3590831403364466,
        1.0137788183007097,
        0.3590831403364466,
        -0.12011232042321396,
        -0.25719315012231203,
    ]
    assert unbiased == [
        -0.025190318327356714,
        -0.011752673231234242,
        0.035100991235234275,
        0.09900183772467866,
        0.035100991235234275,
        -0.011752673231234242,
        -0.025190318327356714,
    ]


def test_cum2x():
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
    y = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)  # Reshape to 2D arrays
    cross_cov = cum2x(x, y, maxlag=50, nsamp=256, overlap=50)
    assert cross_cov.shape == (101, 1)
