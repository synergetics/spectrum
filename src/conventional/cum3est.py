#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from tools import *


def cum3est(y, maxlag, nsamp, overlap, flag, k1):
  """
  UM3EST Third-order cumulants.
  Should be invoked via "CUMEST" for proper parameter checks
  Parameters:
           y: input data vector (column)
      maxlag: maximum lag to be computed
    samp_seg: samples per segment
     overlap: percentage overlap of segments
       flag : 'biased', biased estimates are computed  [default]
              'unbiased', unbiased estimates are computed.
          k1: the fixed lag in c3(m,k1): see below

  Output:
       y_cum:  estimated third-order cumulant,
               C3(m,k1)  -maxlag <= m <= maxlag
  """

  (n1,n2)  = np.shape(y)
  N        = n1*n2
  minlag   = -maxlag
  overlap  = np.fix(overlap/100 * nsamp)
  nrecord  = np.fix((N - overlap)/(nsamp - overlap))
  nadvance = nsamp - overlap

  y_cum = np.zeros([maxlag-minlag+1,1])

  nd = np.arange(nsamp).T
  nlags = 1*maxlag + 1
  zlag = 1 + maxlag
  if flag == 'biased':
    scale = np.ones([nlags, 1])/nsamp
  else:
    lsamp = nsamp - abs(k1)
    scale = make_arr((range(lsamp-maxlag, lsamp), range(lsamp, lsamp-maxlag,-1)), axis=1)
    (m2,n2) = scale.shape
    scale = np.ones([m2,n2]) / scale

  y = y.ravel(order='F')
  for i in xrange(nrecord):
    x = y[ind]
    x = x.ravel(order='F') - mean(x)
    cx = np.conj(x)
    z = x * 0

    # create the "IV" matrix: offset for second lag
    if k1 > 0:
      z[q:nsamp-k1] = x[0:nsamp-k1, :] * cx[k1:nsamp, :]
    else:
      z[-k1:nsamp] = x[-k1:nsamp] * cx[0:nsamp+k1]

    # compute third-order cumulants
    y_cum[zlag] = y_cum[zlag] + (z.T * x)

    for k in xrange(maxlag):
      y_cum[zlag-k] = y_cum[zlag-k] + z[k:nsamp].T * x[0:nsamp-k]
      y_cum[zlag+k] = y_cum[zlag+k] + z[0:nsamp-k].T * x[k:nsamp]

    ind = ind + int(nadvance)

  y_cum = y_cum * scale/nrecord

  return y_cum

