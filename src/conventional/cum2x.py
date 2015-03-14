#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from tools import *
from cum2est import *
from cum3est import *
from cum4est import *


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
  zlag  = maxlag + 1
  y_cum = zeros(nlags,1)

  if flag == 'biased':
    scale = np.ones([nlags, 1])/nsamp
  else:
    scale = make_arr((range(lx-maxlag, lx-1, -1), range(lx-1, lx-maxlag, -1)), axis=1).T
    scale = np.ones([2*maxlag+1, 1]) / scale


  ind = np.arange(nsamp).T
  for k in xrange(nrecs):
    xs = x[ind]
    xs = xs.reshape(1,-1) - np.mean(xs)
    ys = y[ind]
    ys = ys.reshape(1,-1) - np.mean(ys)

    y_cum[zlag] = y_cum[zlag] + xs.T * ys

    for m in xrange(maxlag):
      y_cum[zlag-m] = y_cum[zlag-m] + xs[m:nsamp].T * ys[0:nsamp-m]
      y_cum[zlag+m] = y_cum[zlag+m] + xs[0:nsamp-m].T * ys[m:nsamp]

    ind = ind + int(nadvance)

  y_cum = y_cum * scale / nrecs

  return y_cum

