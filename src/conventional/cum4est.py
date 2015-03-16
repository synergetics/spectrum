#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *
from cum2est import *
from cum2x import *


def cum4est(y, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0, k2=0):
  """
  CUM4EST Fourth-order cumulants.
  Parameters:
          Should be invoked via CUMEST for proper parameter checks
          y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)
          Computes sample estimates of fourth-order cumulants
          via the overlapped segment method.
          y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)
                 y: input data vector (column)
            maxlag: maximum lag
          samp_seg: samples per segment
           overlap: percentage overlap of segments
             flag : 'biased', biased estimates are computed
                  : 'unbiased', unbiased estimates are computed.
      k1,k2 : the fixed lags in C3(m,k1) or C4(m,k1,k2)

  Output:
      y_cum : estimated fourth-order cumulant slice
              C4(m,k1,k2)  -maxlag <= m <= maxlag
  """

  (n1, n2) = shape(y, 2)
  N = n1*n2
  overlap0 = overlap
  overlap = np.fix(overlap/100 * nsamp)
  nrecord = np.fix((N - overlap)/(nsamp - overlap))
  nadvance = nsamp - overlap

  # scale factors for unbiased estimates
  nlags = 2 * maxlag + 1
  zlag  = maxlag
  tmp   = np.zeros([nlags,1])
  if flag == 'biased':
    scale = np.ones([nlags,1])/nsamp
  else:
    ind = np.arange(-maxlag, maxlag+1).T
    kmin = min(0, min(k1, k2))
    kmax  = max(0,max(k1, k2))
    scale = nsamp - np.maximum(ind, kmax) + np.minimum(ind, kmin)
    scale = np.ones(nlags) / scale
    scale = scale.reshape(-1,1)

  mlag  = maxlag + max(abs(k1), abs(k2))
  mlag  = max(mlag, abs(k1-k2) )
  mlag1 = mlag + 1
  nlag  = maxlag
  m2k2  = np.zeros([2*maxlag+1,1])

  if np.any(np.any(np.imag(y) != 0)): complex_flag = 1
  else: complex_flag = 0

  # estimate second- and fourth-order moments combine
  y_cum  = np.zeros([2*maxlag+1, 1])
  R_yy   = np.zeros([2*mlag+1, 1])

  ind = np.arange(nsamp)
  for i in xrange(nrecord):
    tmp = np.zeros([2*maxlag+1, 1])
    x = y[ind]
    x = x.ravel(order='F') - np.mean(x)
    z =  x * 0
    cx = np.conj(x)

    # create the "IV" matrix: offset for second lag
    if k1 >= 0:
      z[0:nsamp-k1] = x[0:nsamp-k1] * cx[k1:nsamp]
    else:
      z[-k1:nsamp] = x[-k1:nsamp] * cx[0:nsamp+k1]

    # create the "IV" matrix: offset for third lag
    if k2 >= 0:
      z[0:nsamp-k2] = z[0:nsamp-k2] * x[k2:nsamp]
      z[nsamp-k2:nsamp] = np.zeros([k2, 1])
    else:
      z[-k2:nsamp] = z[-k2:nsamp] * x[0:nsamp+k2]
      z[0:-k2] = np.zeros([-k2, 1])

    tmp[zlag] = tmp[zlag] + np.dot(z.T, x)

    for k in xrange(1, maxlag+1):
      tmp[zlag-k] = tmp[zlag-k] + np.dot(z[k:nsamp].T, x[0:nsamp-k])
      tmp[zlag+k] = tmp[zlag+k] + np.dot(z[0:nsamp-k].T, x[k:nsamp])

    y_cum = y_cum + tmp * scale

    R_yy = cum2est(x, mlag, nsamp, overlap0, flag)
    #  We need E x(t)x(t+tau) stuff also:
    if complex_flag:
      M_yy = cum2x(np.conj(x), x, mlag, nsamp, overlap0, flag)
    else:
      M_yy = R_yy

    y_cum = y_cum - \
            R_yy[mlag1+k1-1] * R_yy[mlag1-k2-nlag-1:mlag1-k2+nlag] - \
            R_yy[k1-k2+mlag1-1] * R_yy[mlag1-nlag-1:mlag1+nlag] - \
            M_yy[mlag1+k2-1].T * M_yy[mlag1-k1-nlag-1:mlag1-k1+nlag]

    ind = ind + int(nadvance)


  y_cum = y_cum / nrecord

  return y_cum

