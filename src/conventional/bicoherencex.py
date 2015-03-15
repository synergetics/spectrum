#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *


def bicoherencex(w, x, y, nfft=None, wind=None, nsamp=None, overlap=None):
  """
  Direct (FD) method for estimating cross-bicoherence
  Parameters:
    w,x,y - data vector or time-series
          - should have identical dimensions
    nfft - fft length [default = power of two > nsamp]
           actual size used is power of two greater than 'nsamp'
    wind - specifies the time-domain window to be applied to each
           data segment; should be of length 'segsamp' (see below);
      otherwise, the default Hanning window is used.
    segsamp - samples per segment [default: such that we have 8 segments]
            - if x is a matrix, segsamp is set to the number of rows
    overlap - percentage overlap, 0 to 99  [default = 50]
            - if y is a matrix, overlap is set to 0.

  Output:
    bic     - estimated cross-bicoherence: an nfft x nfft array, with
              origin at center, and axes pointing down and to the right.
    waxis   - vector of frequencies associated with the rows and columns
              of bic;  sampling frequency is assumed to be 1.
  """

  if w.shape != x.shape or x.shape != y.shape:
    raise ValueError('w, x and y should have identical dimentions')

  (ly, nrecs) = y.shape
  if ly == 1:
    ly = nrecs
    nrecs = 1
    w = w.reshape(1,-1)
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)

  if not nfft:
    nfft = 128

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

  if not wind:
    wind = np.hanning(nsamp)

  try:
    (rw, cw) = wind.shape
  except ValueError:
    (rw,) = wind.shape
    cw = 1

  if min(rw, cw) != 1 or max(rw, cw) != nsamp:
    print "Segment size is " + str(nsamp)
    print "Wind array is " + str(rw) + " by " + str(cw)
    print "Using default Hanning window"
    wind = np.hanning(nsamp)

  wind = wind.reshape(1,-1)


  # Accumulate triple products
  bic = np.zeros([nfft, nfft])
  Pyy = np.zeros([nfft,1])
  Pww = np.zeros([nfft,1])
  Pxx = np.zeros([nfft,1])

  mask = hankel(np.arange(nfft),np.array([nfft-1]+range(nfft-1)))
  Yf12 = np.zeros([nfft,nfft])
  ind  = np.transpose(np.arange(nsamp))
  w = w.ravel(order='F')
  x = x.ravel(order='F')
  y = y.ravel(order='F')

  for k in xrange(nrecs):
    ws = w[ind]
    ws = (ws - np.mean(ws)) * wind
    Wf = np.fft.fft(ws, nfft) / nsamp
    CWf = np.conjugate(Wf)
    Pww = Pww + flat_eq(Pww, (Wf*CWf))

    xs = x[ind]
    xs = (xs - np.mean(xs)) * wind
    Xf = np.fft.fft(xs, nfft) / nsamp
    CXf = np.conjugate(Xf)
    Pxx = Pxx + flat_eq(Pxx, (Xf*CXf))

    ys = y[ind]
    ys = (ys - np.mean(ys)) * wind
    Yf = np.fft.fft(ys, nfft) / nsamp
    CYf = np.conjugate(Yf)
    Pyy = Pyy + flat_eq(Pyy, (Yf*CYf))

    Yf12 = flat_eq(Yf12, CYf.ravel(order='F')[mask])
    bic = bic + (Wf * np.transpose(Xf)) * Yf12

    ind = ind + int(nadvance)

  bic = bic / nrecs
  Pww = Pww / nrecs
  Pxx = Pxx / nrecs
  Pyy = Pyy / nrecs
  mask = flat_eq(mask, Pyy.ravel(order='F')[mask])

  bic = abs(bic)**2 / ((Pww * np.transpose(Pxx)) * mask)
  bic = np.fft.fftshift(bic)

  # Contour plot of magnitude bispectrum
  if nfft%2 == 0:
    waxis = np.transpose(np.arange(-1*nfft/2, nfft/2)) / nfft
  else:
    waxis = np.transpose(np.arange(-1*(nfft-1)/2, (nfft-1)/2+1)) / nfft

  cont = plt.contourf(waxis,waxis,bic,100, cmap=plt.cm.Spectral_r)
  plt.colorbar(cont)
  plt.title('Bicoherence estimated via the direct (FFT) method')
  plt.xlabel('f1')
  plt.ylabel('f2')

  colmax, row = bic.max(0), bic.argmax(0)
  maxval, col = colmax.max(0), colmax.argmax(0)
  print 'Max: bic('+str(waxis[col])+','+str(waxis[col])+') = '+str(maxval)
  plt.show()

  return (bic, waxis)


def test():
  nl1 = sio.loadmat(here(__file__) + '/demo/nl1.mat')
  dbic = bicoherencex(nl1['x'], nl1['x'], nl1['y'])


if __name__ == '__main__':
  test()
