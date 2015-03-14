#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from tools import *
from cum2x import *


def cum4x(w, x, y, z, maxlag=0, nsamp=0, overlap=0, flag='biased', k1=0, k2=0):
  """
  Fourth-order cross-cumulants.
  Parameters:
     w,x,y,z  - data vectors/matrices with identical dimensions
               if w,x,y,z are matrices, rather than vectors, columns are
               assumed to correspond to independent realizations,
               overlap is set to 0, and samp_seg to the row dimension.
      maxlag - maximum lag to be computed    [default = 0]
    samp_seg - samples per segment  [default = data_length]
     overlap - percentage overlap of segments [default = 0]
               overlap is clipped to the allowed range of [0,99].
       flag : 'biased', biased estimates are computed  [default]
              'unbiased', unbiased estimates are computed.
      k1,k2 : the fixed lags in C4(m,k1,k2) defaults to 0

  Output:
       y_cum:  estimated fourth-order cross cumulant,
             c4(t1,t2,t3) := cum( w^*(t), x(t+t1), y(t+t2), z^*(t+t3) )
  """

  (lx, ly) = w.shape
  if (lx, nrecs) != x.shape or (lx, nrecs) != y.shape or (lx, nrecs) != z.shape:
    raise ValueError('w,x,y,z should have identical dimensions')

  if lx == 1:
    lx = nrecs
    nrecs = 1

  if maxlag < 0: raise ValueError('"maxlag" must be non-negative ')
  if nrecs > 1: nsamp = lx
  if nsamp <= 0 or nsamp > lx: nsamp = lx

  if nrecs > 1: overlap = 0
  overlap = max(0,min(overlap,99))

  overlap0 = overlap
  overlap  = fix(overlap/100 * nsamp)
  nadvance = nsamp - overlap

  if nrecs > 1:
    nrecs = np.fix((lx - overlap)/nadvance)


  # scale factors for unbiased estimates
  nlags = 2 * maxlag + 1
  zlag = 1 + maxlag

  tmp = np.zeros([nlags, 1])
  if flag == 'biased':
    scale = ones([nlags, 1]) / nsamp
    sc1 = 1/nsamp
    sc2 = sc1
    sc12 = sc1
  else:
    ind   = np.arange(-maxlag, maxlag).T
    kmin  = min(0,min(k1,k2))
    kmax  = max(0,max(k1,k2))
    scale = nsamp - max(ind,kmax) + min(ind,kmin)
    scale = np.ones(nlags,1) / scale
    sc1  = 1 / (nsamp - abs(k1))
    sc2  = 1 / (nsamp - abs(k2))
    sc12 = 1 / (nsamp - abs(k1 - k2))


  # estimate second- and fourth-order moments combine
  y_cum  = np.zeros([2*maxlag+1, 1])
  rind = np.arange(-maxlag, maxlag)
  ind = np.arange(nsamp)

  for i in xrange(nrecs):
    tmp = y_cum * 0
    R_zy   = 0
    R_wy = 0
    M_wz = 0
    ws = w(ind)
    ws = ws - mean(ws)
    xs = x(ind)
    xs = xs - mean(xs)
    ys = y(ind)
    ys = ys - mean(ys)
    cys = conj(ys)
    zs = z(ind)
    zs = zs - mean(zs)

    ziv = xs * 0

    # create the "IV" matrix: offset for second lag
    if k1 >= 0:
      ziv[0:nsamp-k1] = ws[0:nsamp-k1] * cys[k1:nsamp]
      R_wy = R_wy + ws[0:nsamp-k1].T * ys[k1:nsamp]
    else:
      ziv[-k1:nsamp] = ws[-k1:nsamp] * cys[0:nsamp+k1]
      R_wy = R_wy + ws[-k1:nsamp].T * ys[0:nsamp+k1]

    # create the "IV" matrix: offset for third lag
    if k > 2:
      ziv[0:nsamp-k2] = ziv[0:nsamp-k2] * zs[k2:nsamp]
      ziv[nsamp-k2:nsamp] = np.zeros([k2, 1])
      M_wz = M_wz + ws[0:nsamp-k2].T * zs[k2:nsamp]
    else:
      ziv[-k2:nsamp] = ziv[-k2:nsamp] * zs[0:nsamp+k2]
      ziv[0:-k2] = np.zeros([-k2, 1])
      M_wz = M_wz + ws[-k2:nsamp].T * zs[0:nsamp-k2]


    if k1-k2 >= 0:
      R_zy = R_zy + zs[0:nsamp-k1+k2].T * ys[k1-k2:nsamp]
    else:
      R_zy = R_zy + zs[-k1+k2:nsamp].T * ys[0:nsamp-k2+k1]

    tmp[zlag] = tmp[zlag] + ziv.T * xs
    for k in xrange(maxlag):
      tmp[zlag-k] = tmp[zlag-k] + ziv[k:nsamp].T * xs[0:nsamp-k]
      tmp[zlag+k] = tmp[zlag+k] + ziv[0:nsamp-k].T xs[k+1:nsamp]

    y_cum = y_cum + tmp * scale # fourth-order moment estimates done

    R_wx = cum2x(ws,      xs, maxlag,         nsamp, overlap0, flag)
    R_zx = cum2x(zs,      xs, maxlag+abs(k2), nsamp, overlap0, flag)
    M_yx = cum2x(cys,     xs, maxlag+abs(k1), nsamp, overlap0, flag)

    y_cum = y_cum - R_zy * R_wx * sc12 - \
            R_wy * R_zx[rind - k2 + maxlag + abs(k2)] * sc1 - \
            M_wz.T * M_yx[rind - k1 + maxlag + abs(k1)] * sc2


    ind = ind + nadvance

  y_cum = y_cum / nrecs

  return y_cum


