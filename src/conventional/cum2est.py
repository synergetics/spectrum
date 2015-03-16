#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *


def cum2est(y, maxlag, nsamp, overlap, flag):
  """
  CUM2EST Covariance function.
  Should be involed via "CUMEST" for proper parameter checks.
  Parameters:
           y: input data vector (column)
      maxlag: maximum lag to be computed
    samp_seg: samples per segment (<=0 means no segmentation)
     overlap: percentage overlap of segments
        flag: 'biased', biased estimates are computed
              'unbiased', unbiased estimates are computed.

  Output:
       y_cum: estimated covariance,
              C2(m)  -maxlag <= m <= maxlag
  """

  (n1, n2) = shape(y, 2)
  N = n1*n2
  overlap  = np.fix(overlap/100 * nsamp)
  nrecord  = np.fix((N - overlap)/(nsamp - overlap))
  nadvance = nsamp - overlap

  y_cum    = np.zeros([maxlag+1, 1])
  ind = np.arange(nsamp)
  y = y.ravel(order='F')

  for i in xrange(nrecord):
    x = y[ind]
    x = x - np.mean(x)

    for k in xrange(maxlag+1):
      y_cum[k] = y_cum[k] + np.dot(x[0:nsamp-k].T, x[k:nsamp])

    ind = ind + int(nadvance)

  if flag == 'biased':
    y_cum = y_cum / (nsamp * nrecord)
  else:
    y_cum = y_cum / (nrecord * (nsamp-np.matrix(range(maxlag+1)).T))
    y_cum = np.asarray(y_cum)

  if maxlag > 0:
    y_cum = make_arr([np.conj(y_cum[maxlag:0:-1]), y_cum], axis=0)

  return y_cum

