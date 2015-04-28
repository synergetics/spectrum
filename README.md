Higher Order Spectrum Estimation toolkit
======

# Contents

1. Bicoherence
```python
from spectrum import bicoherence

bicoherence(y, nfft=None, wind=None, nsamp=None, overlap=None)
```
![bicoher](https://raw.githubusercontent.com/synergetics/spectrum/master/images/bicoherence.png)

- Cross Bicoherence
```python
from spectrum import bicoherencex

bicoherencex(w, x, y, nfft=None, wind=None, nsamp=None, overlap=None)
```
![bicoherx](https://raw.githubusercontent.com/synergetics/spectrum/master/images/cross_bicoherence.png)

- Bispectrum Direct (using fft)
```python
from spectrum import bispectrumd

bispectrumd(y, nfft=None, wind=None, nsamp=None, overlap=None)
```
![bispectr](https://raw.githubusercontent.com/synergetics/spectrum/master/images/bispectrumd.png)

- Bispectrum Indirect
```python
from spectrum import bispectrumi

bispectrumi(y, nlag=None, nsamp=None, overlap=None, flag='biased', nfft=None, wind=None)
```
![bispectri](https://raw.githubusercontent.com/synergetics/spectrum/master/images/bispectrum_indirect.png)

- Cross Bispectrum (Direct)
```python
from spectrum import bispectrumdx

bispectrumdx(x, y, z, nfft=None, wind=None, nsamp=None, overlap=None)
```
![bispectrdx](https://raw.githubusercontent.com/synergetics/spectrum/master/images/cross_bispectrum.png)

- 2nd, 3rd and 4th order cumulants
```python
from spectrum import cumest

order = 2 # 2nd order
cumest(y, norder=order, maxlag=0 ,nsamp=None, overlap=0, flag='biased' ,k1=0, k2=0)
```

- 2nd, 3rd and 4th order cross-cumulants
```python
from spectrum import cum2x, cum3x, cum4x

cum2x(x, y, maxlag=0, nsamp=0, overlap=0, flag='biased')
cum3x(x, y, z, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0)
cum4x(w, x, y, z, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0, k2=0)
```

More coming soon

Taken from the Higher Order Spectral Analysis toolkit for MATLAB

