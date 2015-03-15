#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
import scipy.io as sio
import matplotlib.pyplot as plt

from ..tools import *
from cum2est import *
from cum3est import *
from cum4est import *


def cumest(y, norder=2, maxlag=0 ,nsamp=None, overlap=0, flag='biased' ,k1=0, k2=0):
  """
  Second-, third- or fourth-order cumulants.
  Parameters:
           y - time-series  - should be a vector
      norder - cumulant order: 2, 3 or 4 [default = 2]
      maxlag - maximum cumulant lag to compute [default = 0]
    samp_seg - samples per segment  [default = data_length]
     overlap - percentage overlap of segments [default = 0]
               overlap is clipped to the allowed range of [0,99].
       flag  - 'biased' or 'unbiased'  [default = 'biased']
      k1,k2  - specify the slice of 3rd or 4th order cumulants

  Output:
      y_cum  - C2(m) or C3(m,k1) or C4(m,k1,k2),  -maxlag <= m <= maxlag
               depending upon the cumulant order selected
  """

  (ksamp, nrecs) = y.shape
  if ksamp == 1:
    ksamp = nrecs
    nrecs = 1

  if norder < 2 or norder > 4:
    raise ValueError('cumulant order must be 2, 3 or 4')

  if maxlag < 0:
    raise ValueError('"maxlag" must be non-negative')

  if nrecs > 1: nsamp = ksamp
  if nsamp <= 0 or nsamp > ksamp: nsamp = ksamp

  if nrecs > 1: overlap = 0
  overlap = max(0,min(overlap,99))


  # estimate the cumulants
  if norder == 2:
    y_cum = cum2est(y, maxlag, nsamp, overlap, flag)
  elif norder == 3:
    y_cum = cum3est(y, maxlag, nsamp, overlap, flag, k1)
  elif norder == 4:
    y_cum = cum3est(y, maxlag, nsamp, overlap, flag, k1, k2)


  return y_cum


