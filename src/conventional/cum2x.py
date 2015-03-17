#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *


def cum2x(x, y, maxlag=0, nsamp=0, overlap=0, flag='biased'):
  """
  Cross-covariance
  Parameters:
      x,y    - data vectors/matrices with identical dimensions
               if x,y are matrices, rather than vectors, columns are
               assumed to correspond to independent realizations,
               overlap is set to 0, and samp_seg to the row dimension.
      maxlag - maximum lag to be computed    [default = 0]
    samp_seg - samples per segment  [default = data_length]
     overlap - percentage overlap of segments [default = 0]
               overlap is clipped to the allowed range of [0,99].
       flag  - 'biased', biased estimates are computed  [default]
              'unbiased', unbiased estimates are computed.

  Output:
       y_cum - estimated cross-covariance
               E x^*(n)y(n+m),   -maxlag <= m <= maxlag
  """

  (lx, nrecs) = x.shape
  if (lx, nrecs) != y.shape:
    raise ValueError('x,y should have identical dimensions')

  if lx == 1:
    lx = nrecs
    nrecs = 1

  if maxlag < 0: raise ValueError('maxlag must be non-negative')
  if nrecs > 1: nsamp = lx
  if nsamp <= 0 or nsamp > lx: nsamp = lx
  if nrecs > 1: overlap = 0
  overlap = max(0,min(overlap,99))

  overlap = np.fix(overlap/100 * nsamp)
  nadvance = nsamp - overlap
  if nrecs == 1:
    nrecs = np.fix((lx - overlap)/nadvance)

  nlags = 2*maxlag+1
  zlag  = maxlag
  y_cum = np.zeros([nlags,1])

  if flag == 'biased':
    scale = np.ones([nlags, 1])/nsamp
  else:
    scale = make_arr((range(lx-maxlag, lx+1), range(lx-1, lx-maxlag-1, -1)), axis=1).T
    scale = np.ones([2*maxlag+1, 1]) / scale

  ind = np.arange(nsamp).T
  for k in xrange(nrecs):
    xs = x[ind].ravel(order='F')
    xs = xs - np.mean(xs)
    ys = y[ind].ravel(order='F')
    ys = ys - np.mean(ys)

    y_cum[zlag] = y_cum[zlag] + np.dot(xs, ys)

    for m in xrange(1, maxlag+1):
      y_cum[zlag-m] = y_cum[zlag-m] + np.dot(xs[m:nsamp].T, ys[0:nsamp-m])
      y_cum[zlag+m] = y_cum[zlag+m] + np.dot(xs[0:nsamp-m].T, ys[m:nsamp])

    ind = ind + int(nadvance)

  y_cum = y_cum * scale / nrecs

  return y_cum


def test():
  y = sio.loadmat(here(__file__) + '/demo/ma1.mat')['y']

  # The right results are:
  #           "biased": [--0.25719  -0.12011   0.35908   1.01378   0.35908  -0.12011  -0.25719]
  #           "unbiased": [-0.025190  -0.011753   0.035101   0.099002   0.035101  -0.011753  -0.025190]
  print cum2x(y, y, 3, 100, 0, "biased")
  print cum2x(y, y, 3, 100, 0, "unbiased")


if __name__ == '__main__':
  test()

