#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *


def cum3x(x, y, z, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0):
  """
  Third-order cross-cumulants.
  Parameters:
      x,y,z  - data vectors/matrices with identical dimensions
               if x,y,z are matrices, rather than vectors, columns are
               assumed to correspond to independent realizations,
               overlap is set to 0, and samp_seg to the row dimension.
      maxlag - maximum lag to be computed    [default = 0]
    samp_seg - samples per segment  [default = data_length]
     overlap - percentage overlap of segments [default = 0]
               overlap is clipped to the allowed range of [0,99].
       flag : 'biased', biased estimates are computed  [default]
              'unbiased', unbiased estimates are computed.
          k1: the fixed lag in c3(m,k1): defaults to 0

  Output:
       y_cum:  estimated third-order cross cumulant,
               E x^*(n)y(n+m)z(n+k1),   -maxlag <= m <= maxlag
  """

  (lx, nrecs) = x.shape
  if (lx, nrecs) != y.shape or (lx, nrecs) != z.shape:
    raise ValueError('x,y,z should have identical dimensions')

  if lx == 1:
    lx = nrecs
    nrecs = 1

  if maxlag < 0: raise ValueError('"maxlag" must be non-negative')
  if nrecs > 1: nsamp = lx
  if nsamp <= 0 or nsamp > lx: nsamp = lx
  if nrecs > 1: overlap = 0
  overlap = max(0,min(overlap,99))

  overlap  = np.fix(overlap/100 * nsamp)
  nadvance = nsamp - overlap

  if nrecs == 1:
    nrecs  = np.fix((lx - overlap)/nadvance)

  nlags = 2*maxlag+1
  zlag = maxlag
  y_cum = np.zeros([nlags,1])

  if flag == 'biased':
    scale = np.ones([nlags, 1]) / nsamp
  else:
    lsamp = lx - abs(k1)
    scale = make_arr((range(lsamp-maxlag, lsamp+1), range(lsamp-1, lsamp-maxlag-1, -1)), axis=1).T
    scale = np.ones([2*maxlag+1, 1]) / scale


  if k1 >= 0:
    indx = np.arange(nsamp-k1).T
    indz = np.arange(k1, nsamp).T
  else:
    indx = np.arange(-k1, nsamp).T
    indz = np.arange(nsamp+k1)

  ind = np.arange(nsamp)

  for k in xrange(nrecs):
    xs = x[ind]
    xs = xs - np.mean(xs)
    ys = y[ind]
    ys = ys - np.mean(ys)
    zs = z[ind]
    zs = zs - np.mean(zs)
    zs = np.conj(zs)

    u = np.zeros([nsamp, 1])
    u[indx] = xs[indx] * zs[indz]

    y_cum[zlag] = y_cum[zlag] + np.dot(u.T, ys)

    for m in xrange(1, maxlag+1):
      y_cum[zlag-m] = y_cum[zlag-m] + np.dot(u[m:nsamp].T, ys[0:nsamp-m])
      y_cum[zlag+m] = y_cum[zlag+m] + np.dot(u[0:nsamp-m].T, ys[m:nsamp])

    ind = ind + int(nadvance)

  y_cum = y_cum * scale / nrecs

  return y_cum


def test():
  y = sio.loadmat(here(__file__) + '/demo/ma1.mat')['y']

  # The right results are:
  #           "biased": [0.36338   0.42762   0.77703   0.84322   0.73021  -0.13123  -0.40743]
  #           "unbiased": [0.035591   0.041841   0.075956   0.082345   0.071379  -0.012840  -0.039905]
  print cum3x(y, y, y, 3, 100, 0, "biased")
  print cum3x(y, y, y, 3, 100, 0, "unbiased")


if __name__ == '__main__':
  test()


