#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import scipy.io as sio
import matplotlib.pyplot as plt

from tools import *

np.set_printoptions(linewidth=120)

def bispectrumi(y, nlag=None, nsamp=None, overlap=None,
  flag='biased', nfft=None, wind=None):
  """
  Parameters:
    y       - data vector or time-series
    nlag    - number of lags to compute [must be specified]
    segsamp - samples per segment    [default: row dimension of y]
    overlap - percentage overlap     [default = 0]
    flag    - 'biased' or 'unbiased' [default is 'unbiased']
    nfft    - FFT length to use      [default = 128]
    wind    - window function to apply:
              if wind=0, the Parzen window is applied (default)
              otherwise the hexagonal window with unity values is applied.

  Output:
    Bspec   - estimated bispectrum  it is an nfft x nfft array
              with origin at the center, and axes pointing down and to the right
    waxis   - frequency-domain axis associated with the bispectrum.
            - the i-th row (or column) of Bspec corresponds to f1 (or f2)
              value of waxis(i).
  """

  (ly, nrecs) = y.shape
  if ly == 1:
    y = y.reshape(1,-1)
    ly = nrecs
    nrecs = 1

  if not overlap: overlap = 0
  overlap = min(99, max(overlap,0))
  if nrecs > 1: overlap = 0
  if not nsamp: nsamp = ly
  if nsamp > ly or nsamp <= 0: nsamp = ly
  if not 'flag': flag = 'biased'
  if not nfft: nfft = 128
  if not wind: wind = 0

  nlag = min(nlag, nsamp-1)
  if nfft < 2*nlag+1:
    nfft = 2^nextpow2(nsamp)


  # create the lag window
  Bspec = np.zeros([nfft, nfft])
  if wind == 0:
    indx = np.array([range(1,nlag+1)]).T
    window = make_arr((1, np.sin(np.pi*indx/nlag) / (np.pi*indx/nlag)), axis=0)
  else:
    window = np.ones([nlag+1,1])
  window = make_arr((window, np.zeros([nlag,1])), axis=0)

  # cumulants in non-redundant region
  overlap  = np.fix(nsamp * overlap / 100)
  nadvance = nsamp - overlap
  nrecord  = np.fix((ly*nrecs - overlap) / nadvance)

  c3 = np.zeros([nlag+1,nlag+1])
  ind = np.arange(nsamp)
  y = y.ravel(order='F')

  s = 0
  for k in xrange(nrecord):
    x = y[ind].ravel(order='F')
    x = x - np.mean(x)
    ind = ind + int(nadvance)

    for j in xrange(nlag+1):
      z = x[range(nsamp-j)] * x[range(j, nsamp)]
      for i in xrange(j,nlag+1):
        Sum = np.dot(z[range(nsamp-i)].T, x[range(i,nsamp)])
        if flag == 'biased': Sum = Sum/nsamp
        else: Sum = Sum / (nsamp-i)
        c3[i,j] = c3[i,j] + Sum

  c3 = c3 / nrecord

  # cumulants elsewhere by symmetry
  c3 = c3 + np.tril(c3,-1).T   # complete I quadrant
  c31 = c3[1:nlag+1, 1:nlag+1]
  c32 = np.zeros([nlag, nlag])
  c33 = np.zeros([nlag, nlag])
  c34 = np.zeros([nlag, nlag])
  for i in xrange(nlag):
    x = c31[i:nlag, i]
    c32[nlag-1-i,0:nlag-i] = x.T
    c34[0:nlag-i, nlag-1-i] = x
    if i+1 < nlag:
      x = np.flipud(x[1:len(x)])
      c33 = c33 + np.diag(x,i+1) + np.diag(x,-(i+1))

  c33  = c33 + np.diag(c3[0, nlag:0:-1])

  cmat = make_arr(
    (make_arr((c33, c32, np.zeros([nlag,1])), axis=1),
    make_arr((make_arr((c34, np.zeros([1,nlag])), axis=0), c3), axis=1)),
    axis=0
  )

  # apply lag-domain window
  wcmat = cmat
  if wind != -1:
    indx = np.arange(-1*nlag, nlag+1).T
    window = window.reshape(-1,1)
    for k in xrange(-nlag, nlag+1):
      wcmat[:, k+nlag] = (cmat[:, k+nlag].reshape(-1,1) * \
              window[abs(indx-k)] * \
              window[abs(indx)] * \
              window[abs(k)]).reshape(-1,)


  # compute 2d-fft, and shift and rotate for proper orientation
  Bspec = np.fft.fft2(wcmat, (nfft, nfft))
  Bspec = np.fft.fftshift(Bspec) # axes d and r; orig at ctr


  if nfft%2 == 0:
    waxis = np.transpose(np.arange(-1*nfft/2, nfft/2)) / nfft
  else:
    waxis = np.transpose(np.arange(-1*(nfft-1)/2, (nfft-1)/2+1)) / nfft

  cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap=plt.cm.Spectral_r)
  plt.colorbar(cont)
  plt.title('Bispectrum estimated via the indirect method')
  plt.xlabel('f1')
  plt.ylabel('f2')
  plt.show()

  return (Bspec, waxis)


def test():
  qpc = sio.loadmat('demo/qpc.mat')
  dbic = bispectrumi(qpc['zmat'],  21, 64, 0, 'unbiased', 128, 1)


if __name__ == '__main__':
  test()

