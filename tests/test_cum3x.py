import scipy.io as sio
import numpy as np
from spectrum import cum3x


def test_specific():
    y = sio.loadmat("./tests/demo/ma1.mat")["y"]

    biased = cum3x(y, y, y, 3, 100, 0, "biased").flatten().tolist()
    unbiased = cum3x(y, y, y, 3, 100, 0, "unbiased").flatten().tolist()

    print(biased)
    print(unbiased)

    assert biased == [
        0.36338204713801936,
        0.42761933511385336,
        0.7770255083877481,
        0.8432175410276452,
        0.7302082225531652,
        -0.13122644276234727,
        -0.40742990193372225,
    ]
    assert unbiased == [
        0.03559079795671101,
        0.04184142222249054,
        0.07595557266742405,
        0.08234546299098097,
        0.07137910288887246,
        -0.0128401607399557,
        -0.03990498549791598,
    ]


def test_cum3x():
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
    y = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    z = np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))
    x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)  # Reshape to 2D arrays
    cum3 = cum3x(x, y, z, maxlag=50, nsamp=256, overlap=50, k1=5)
    assert cum3.shape == (101, 1)
