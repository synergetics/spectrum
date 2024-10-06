# Higher Order Spectral Analysis Toolkit

This package provides a comprehensive set of tools for higher-order spectral
analysis in Python. It includes functions for estimating bicoherence,
bispectrum, and various orders of cumulants.

## Installation

You can install the HOSA toolkit using pip:

```bash
pip install higher-spectrum
```

## Contents

### Bicoherence

```python
from spectrum import bicoherence, plot_bicoherence

bic, waxis = bicoherence(y, nfft=None, window=None, nsamp=None, overlap=None)
plot_bicoherence(bic, waxis)
```

![bicoher](https://raw.githubusercontent.com/synergetics/spectrum/master/images/bicoherence.png)

### Cross Bicoherence

```python
from spectrum import bicoherencex, plot_cross_bicoherence

bic, waxis = bicoherencex(w, x, y, nfft=None, window=None, nsamp=None, overlap=None)
plot_cross_bicoherence(bic, waxis)
```

![bicoherx](https://raw.githubusercontent.com/synergetics/spectrum/master/images/cross_bicoherence.png)

### Bispectrum Direct (using FFT)

```python
from spectrum import bispectrumd, plot_bispectrum

Bspec, waxis = bispectrumd(y, nfft=None, window=None, nsamp=None, overlap=None)
plot_bispectrum(Bspec, waxis)
```

![bispectr](https://raw.githubusercontent.com/synergetics/spectrum/master/images/bispectrumd.png)

### Bispectrum Indirect

```python
from spectrum import bispectrumi, plot_bispectrum_indirect

Bspec, waxis = bispectrumi(y, nlag=None, nsamp=None, overlap=None, flag='biased', nfft=None, wind='parzen')
plot_bispectrum_indirect(Bspec, waxis)
```

![bispectri](https://raw.githubusercontent.com/synergetics/spectrum/master/images/bispectrum_indirect.png)

### Cross Bispectrum (Direct)

```python
from spectrum import bispectrumdx, plot_cross_bispectrum

Bspec, waxis = bispectrumdx(x, y, z, nfft=None, window=None, nsamp=None, overlap=None)
plot_cross_bispectrum(Bspec, waxis)
```

![bispectrdx](https://raw.githubusercontent.com/synergetics/spectrum/master/images/cross_bispectrum.png)

### Cumulants (2nd, 3rd, and 4th order)

```python
from spectrum import cumest, plot_cumulant

order = 2  # 2nd order
y_cum = cumest(y, norder=order, maxlag=20, nsamp=None, overlap=0, flag='biased', k1=0, k2=0)
plot_cumulant(np.arange(-20, 21), y_cum, order)
```

### Cross-Cumulants (2nd, 3rd, and 4th order)

```python
from spectrum import cum2x, cum3x, cum4x, plot_cross_covariance, plot_third_order_cross_cumulant, plot_fourth_order_cross_cumulant

# 2nd order cross-cumulant
ccov = cum2x(x, y, maxlag=20, nsamp=None, overlap=0, flag='biased')
plot_cross_covariance(np.arange(-20, 21), ccov)

# 3rd order cross-cumulant
c3 = cum3x(x, y, z, maxlag=20, nsamp=None, overlap=0, flag='biased', k1=0)
plot_third_order_cross_cumulant(np.arange(-20, 21), c3, k1=0)

# 4th order cross-cumulant
c4 = cum4x(w, x, y, z, maxlag=20, nsamp=None, overlap=0, flag='biased', k1=0, k2=0)
plot_fourth_order_cross_cumulant(np.arange(-20, 21), c4, k1=0, k2=0)
```

## Features

- Estimation of bicoherence and cross-bicoherence
- Direct and indirect methods for bispectrum estimation
- Cross-bispectrum estimation
- Cumulant estimation up to 4th order
- Cross-cumulant estimation up to 4th order
- Plotting functions for all estimations

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

## Contributing

Contributions to the HOSA toolkit are welcome! Please feel free to submit a Pull
Request.

## License

This project is licensed under the MIT License.

## Acknowledgements

This toolkit is based on the Higher Order Spectral Analysis toolkit for MATLAB.
We've adapted and extended it for Python users.
