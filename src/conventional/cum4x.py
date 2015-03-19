#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *
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

  (lx, nrecs) = w.shape
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
  overlap  = np.fix(overlap/100 * nsamp)
  nadvance = nsamp - overlap

  if nrecs == 1:
    nrecs = np.fix((lx - overlap)/nadvance)


  # scale factors for unbiased estimates
  nlags = 2 * maxlag + 1
  zlag = maxlag

  tmp = np.zeros([nlags, 1])
  if flag == 'biased':
    scale = np.ones([nlags, 1]) / nsamp
    sc1 = 1/nsamp
    sc2 = sc1
    sc12 = sc1
  else:
    ind   = np.arange(-maxlag, maxlag+1).T
    kmin  = min(0,min(k1,k2))
    kmax  = max(0,max(k1,k2))
    scale = nsamp - np.maximum(ind,kmax) + np.minimum(ind,kmin)
    scale = np.ones(nlags) / scale
    sc1  = 1 / (nsamp - abs(k1))
    sc2  = 1 / (nsamp - abs(k2))
    sc12 = 1 / (nsamp - abs(k1 - k2))
    scale = scale.reshape(-1,1)

  # estimate second- and fourth-order moments combine
  y_cum  = np.zeros([2*maxlag+1, 1])
  rind = np.arange(-maxlag, maxlag+1)
  ind = np.arange(nsamp)

  print nrecs
  for i in xrange(nrecs):
    tmp = y_cum * 0
    R_zy   = 0
    R_wy = 0
    M_wz = 0
    ws = w[ind]
    ws = ws - np.mean(ws)
    xs = x[ind]
    xs = xs - np.mean(xs)
    ys = y[ind]
    ys = ys - np.mean(ys)
    cys = np.conj(ys)
    zs = z[ind]
    zs = zs - np.mean(zs)

    ziv = xs * 0

    # create the "IV" matrix: offset for second lag
    if k1 >= 0:
      ziv[0:nsamp-k1] = ws[0:nsamp-k1] * cys[k1:nsamp]
      R_wy = R_wy + np.dot(ws[0:nsamp-k1].T, ys[k1:nsamp])
    else:
      ziv[-k1:nsamp] = ws[-k1:nsamp] * cys[0:nsamp+k1]
      R_wy = R_wy + np.dot(ws[-k1:nsamp].T, ys[0:nsamp+k1])

    # create the "IV" matrix: offset for third lag
    if k2 > 2:
      ziv[0:nsamp-k2] = ziv[0:nsamp-k2] * zs[k2:nsamp]
      ziv[nsamp-k2:nsamp] = np.zeros([k2, 1])
      M_wz = M_wz + np.dot(ws[0:nsamp-k2].T, zs[k2:nsamp])
    else:
      ziv[-k2:nsamp] = ziv[-k2:nsamp] * zs[0:nsamp+k2]
      ziv[0:-k2] = np.zeros([-k2, 1])
      M_wz = M_wz + np.dot(ws[-k2:nsamp].T, zs[0:nsamp-k2])


    if k1-k2 >= 0:
      R_zy = R_zy + np.dot(zs[0:nsamp-k1+k2].T, ys[k1-k2:nsamp])
    else:
      R_zy = R_zy + np.dot(zs[-k1+k2:nsamp].T, ys[0:nsamp-k2+k1])

    tmp[zlag] = tmp[zlag] + np.dot(ziv.T, xs)
    for k in xrange(1, maxlag+1):
      tmp[zlag-k] = tmp[zlag-k] + np.dot(ziv[k:nsamp].T, xs[0:nsamp-k])
      tmp[zlag+k] = tmp[zlag+k] + np.dot(ziv[0:nsamp-k].T, xs[k:nsamp])

    print y_cum.shape
    y_cum = y_cum + tmp * scale # fourth-order moment estimates done
    print y_cum.shape

    R_wx = cum2x(ws,      xs, maxlag,         nsamp, overlap0, flag)
    R_zx = cum2x(zs,      xs, maxlag+abs(k2), nsamp, overlap0, flag)
    M_yx = cum2x(cys,     xs, maxlag+abs(k1), nsamp, overlap0, flag)

    y_cum = y_cum - R_zy * R_wx * sc12 - \
            R_wy * R_zx[rind - k2 + maxlag + abs(k2)] * sc1 - \
            M_wz.T * M_yx[rind - k1 + maxlag + abs(k1)] * sc2

    ind = ind + int(nadvance)

  y_cum = y_cum / nrecs

  return y_cum



def test():

  y = sio.loadmat(here(__file__) + '/demo/ma1.mat')['y']

  # The right results are:
  #           "biased": [-0.52343  -0.43057   1.16651   3.21583   1.98088  -0.38022  -1.05836]
  #           "unbiased": [-0.53962  -0.43936   1.17829   3.21583   2.00089  -0.38798  -1.09109]
  print cum4x(y, y, y, y, 3, 100, 0, "biased")
  print cum4x(y, y, y, y, 3, 100, 0, "unbiased")


if __name__ == '__main__':
  test()



