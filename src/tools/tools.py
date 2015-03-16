#!/usr/bin/env python


import numpy as np
import sys
import os


def nextpow2(num):
  '''
  Returns the next highest power of 2 from the given value.
  Example
  -------
  >>>nextpow2(1000)
  1024
  >>nextpow2(1024)
  2048

  Taken from: https://github.com/alaiacano/frfft/blob/master/frfft.py
  '''

  npow = 2
  while npow <= num:
      npow = npow * 2
  return npow


def flat_eq(x, y):
  """
  Emulate MATLAB's assignment of the form
  x(:) = y
  """
  z = x.reshape(1, -1)
  z = y
  return z.reshape(x.shape)


def make_arr(arrs, axis=0):
  """
  Create arrays like MATLAB does
  python                                 MATLAB
  make_arr((4, range(1,10)), axis=0) => [4; 1:9]
  """
  a = []
  ctr = 0
  for x in arrs:
    if len(np.shape(x)) == 0:
      a.append(np.array([[x]]))
    elif len(np.shape(x)) == 1:
      a.append(np.array([x]))
    else:
      a.append(x)
    ctr += 1
  return np.concatenate(a, axis)


def shape(o, n):
  """
  Behave like MATLAB's shape
  """
  s = o.shape
  if len(s) < n:
    x = tuple(np.ones(n-len(s)))
    return s + x
  else:
    return s


def here(f=__file__):
  """
  This script's directory
  """
  return os.path.dirname(os.path.realpath(f))

