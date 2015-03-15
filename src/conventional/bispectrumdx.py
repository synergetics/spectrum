#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *


def bispectrumdx(x, y, z, nfft=None, wind=None, nsamp=None, overlap=None):
  """
  Parameters:
    x    - data vector or time-series
    y    - data vector or time-series  (same dimensions as x)
    z    - data vector or time-series  (same dimensions as x)
    nfft - fft length [default = power of two > segsamp]
    wind - window specification for frequency-domain smoothing
           if 'wind' is a scalar, it specifies the length of the side
              of the square for the Rao-Gabr optimal window  [default=5]
           if 'wind' is a vector, a 2D window will be calculated via
              w2(i,j) = wind(i) * wind(j) * wind(i+j)
           if 'wind' is a matrix, it specifies the 2-D filter directly
    segsamp - samples per segment [default: such that we have 8 segments]
            - if x is a matrix, segsamp is set to the number of rows
    overlap - percentage overlap, allowed range [0,99]. [default = 50];
            - if x is a matrix, overlap is set to 0.

  Output:
    Bspec   - estimated bispectrum: an nfft x nfft array, with origin
              at the center, and axes pointing down and to the right.
    waxis   - vector of frequencies associated with the rows and columns
              of Bspec;  sampling frequency is assumed to be 1.
  """

  (lx, lrecs) = x.shape
  (ly, nrecs) = y.shape
  (lz, krecs) = z.shape

  if lx != ly or lrecs != nrecs or ly != lz or nrecs != krecs:
    raise Exception('x, y and z should have identical dimensions')

  if ly == 1:
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)
    z = z.reshape(1,-1)
    ly = nrecs
    nrecs = 1

  if not overlap: overlap = 50
  overlap = max(0,min(overlap,99))
  if nrecs > 1: overlap = 0
  if not nsamp: nsamp = 0
  if nrecs > 1: nsamp = ly
  if nrecs == 1 and nsamp <= 0:
    nsamp = np.fix(ly/ (8 - 7 * overlap/100))
  if nfft < nsamp:
    nfft = 2**nextpow2(nsamp)

  overlap = np.fix(overlap/100 * nsamp)
  nadvance = nsamp - overlap
  nrecs = np.fix((ly*nrecs - overlap) / nadvance)


  # create the 2-D window
  if not wind: wind = 5

  m = n = 0
  try:
    (m, n) = wind.shape
  except ValueError:
    (m,) = wind.shape
    n = 1
  except AttributeError:
    m = n = 1

  window = wind
  # scalar: wind is size of Rao-Gabr window
  if max(m, n) == 1:
    winsize = wind
    if winsize < 0: winsize = 5 # the window size L
    winsize = winsize - (winsize%2) + 1 # make it odd
    if winsize > 1:
      mwind = np.fix(nfft/winsize) # the scale parameter M
      lby2 = (winsize - 1)/2

      theta = np.array([np.arange(-1*lby2, lby2+1)]) # force a 2D array
      opwind = np.ones([winsize, 1]) * (theta**2) # w(m,n) = m**2
      opwind = opwind + opwind.transpose() + (np.transpose(theta) * theta) # m**2 + n**2 + mn
      opwind = 1 - ((2*mwind/nfft)**2) * opwind
      Hex = np.ones([winsize,1]) * theta
      Hex = abs(Hex) + abs(np.transpose(Hex)) + abs(Hex + np.transpose(Hex))
      Hex = (Hex < winsize)
      opwind = opwind * Hex
      opwind = opwind * (4 * mwind**2) / (7 * np.pi**2)
    else:
      opwind = 1

  # 1-D window passed: convert to 2-D
  elif min(m, n) == 1:
    window = window.reshape(1,-1)

    if np.any(np.imag(window)) != 0:
      print "1-D window has imaginary components: window ignored"
      window = 1

    if np.any(window) < 0:
      print "1-D window has negative components: window ignored"
      window = 1

    lwind = np.size(window)
    w = window.ravel(order='F')
    # the full symmetric 1-D
    windf = np.array(w[range(lwind-1, 0, -1) + [window]])
    window = np.array([window], np.zeros([lwind-1,1]))
    # w(m)w(n)w(m+n)
    opwind = (windf * np.transpose(windf)) * hankel(np.flipud(window), window)
    winsize = np.size(window)

  # 2-D window passed: use directly
  else:
    winsize = m

    if m != n:
      print "2-D window is not square: window ignored"
      window = 1
      winsize = m

    if m%2 == 0:
      print "2-D window does not have odd length: window ignored"
      window = 1
      winsize = m

    opwind = window

  # accumulate triple products
  Bspec = np.zeros([nfft, nfft]) # the hankel mask (faster)
  mask = hankel(np.arange(nfft),np.array([nfft-1]+range(nfft-1)))
  locseg = np.arange(nsamp).transpose()
  x = x.ravel(order='F')
  y = y.ravel(order='F')
  z = z.ravel(order='F')

  for krec in xrange(nrecs):
    xseg = x[locseg].reshape(1,-1)
    yseg = y[locseg].reshape(1,-1)
    zseg = z[locseg].reshape(1,-1)

    Xf = np.fft.fft(xseg - np.mean(xseg), nfft) / nsamp
    Yf = np.fft.fft(yseg - np.mean(yseg), nfft) / nsamp
    CZf = np.fft.fft(zseg - np.mean(zseg), nfft) / nsamp
    CZf = np.conjugate(CZf).ravel(order='F')

    Bspec = Bspec + \
      flat_eq(Bspec, (Xf * np.transpose(Yf)) * CZf[mask].reshape(nfft, nfft))
    locseg = locseg + int(nadvance)

  Bspec = np.fft.fftshift(Bspec) / nrecs


  # frequency-domain smoothing
  if winsize > 1:
    lby2 = int((winsize-1)/2)
    Bspec = convolve2d(Bspec,opwind)
    Bspec = Bspec[range(lby2+1,lby2+nfft+1), :][:, np.arange(lby2+1,lby2+nfft+1)]


  if nfft%2 == 0:
    waxis = np.transpose(np.arange(-1*nfft/2, nfft/2)) / nfft
  else:
    waxis = np.transpose(np.arange(-1*(nfft-1)/2, (nfft-1)/2+1)) / nfft

  # cont1 = plt.contour(abs(Bspec), 4, waxis, waxis)
  cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap=plt.cm.Spectral_r)
  plt.colorbar(cont)
  plt.title('Bispectrum estimated via the direct (FFT) method')
  plt.xlabel('f1')
  plt.ylabel('f2')
  plt.show()

  return (Bspec, waxis)


def test():
  nl1 = sio.loadmat(here(__file__) + '/demo/nl1.mat')
  dbic = bispectrumdx(nl1['x'], nl1['x'], nl1['y'], 128,5)


if __name__ == '__main__':
  test()

