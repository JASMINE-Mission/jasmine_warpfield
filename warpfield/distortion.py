#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from functools import reduce
from operator import add
from scipy.optimize import least_squares
import numpy as np


def distortion_generator(K=[0,],S=[0,0],T=[0,],scale=1.0):
  ''' Generate a distortion function with distortion parameters.

  Parameters:
    K (list): radial distortion parameters.
    S (list): tangential distortion parameters.
    T (list): tangential distortion parameters.

  Return:
    A distortion function.
  '''
  K = np.array(K)
  if K.ndim==0: K = np.expand_dims(K, axis=0)
  S = np.array(S).reshape((2,1))
  T = np.array(T)
  if T.ndim==0: T = np.expand_dims(T, axis=0)

  def distortion(position: np.ndarray):
    ''' Generated distortion function.

    Parameters:
      position (ndarray):
          A numpy.array with the shape of (2, Nsrc). The first element
          contains the x-positions, while the second element contains
          the y-positions.

    Return:
      A numpy.ndarray of the input coordinates.
    '''
    position = np.array(position)
    xy = np.expand_dims(position.prod(axis=0),axis=0)
    r = np.sqrt(np.square(position).sum(axis=0))
    Kr = 1+reduce(add,[k*(r/1e6)**(2+2*n) for n,k in enumerate(K)])
    Tr = 1+reduce(add,[t*(r/1e6)**(2+2*n) for n,t in enumerate(T)])
    Px = (np.diag(S.flat)@(r**2+2*position**2) + 2*S[::-1,:]*xy)/5e6
    return scale*(position*Kr + Px*Tr)

  return distortion


@dataclass
class Sip(object):
  ''' Simple Imaging Polynomical convention.

  Attributes:
    order (int):
        The polynomial order of the SIP convention.
    A (ndarray):
        The SIP coefficient matrix for the x-coordinate.
        The shape of the matrix should be (order+1, order+1).
    B (ndarray):
        The SIP coefficient matrix for the y-coordinate.
        The shape of the matrix should be (order+1, order+1).
  '''
  order: int
  center: np.ndarray = field(init=False)
  A: np.ndarray
  B: np.ndarray

  def __post_init__(self):
    self.center = np.array((0,0))
    assert self.order >= 0, \
      f'The polynomical order should be non-negative.'
    assert self.A.shape == (self.order+1,self.order+1), \
      f'The shape of A matris should be ({self.order+1}, {self.order+1}).'
    assert self.B.shape == (self.order+1,self.order+1), \
      f'The shape of B matris should be ({self.order+1}, {self.order+1}).'

  def apply(self, position: np.ndarray):
    ''' Modify xy-coordinates with the SIP function.

    Parameters:
      position (ndarray):
          An array contains the list of coordinates. The shape of the array
          should be (2, Nsrc), where Nsrc is the number of sources.

    Return:
      An ndarray instance contains modified coordinates.
    '''
    position = np.array(position).reshape((2,-1))
    N = position.shape[1]
    cx,cy = self.center
    x,y = position[0]-cx, position[1]-cy

    dx = np.zeros_like(x)
    tmp = np.zeros((self.order+1,N))
    for m in np.arange(self.order+1):
      n = self.order+1 - m
      tmp[m] = np.sum([self.A[m,i]*y**i for i in np.arange(n)],axis=0)
    for m in np.arange(self.order+1):
      dx += tmp[m]*x**m

    dy = np.zeros_like(x)
    tmp = np.zeros((self.order+1,N))
    for m in np.arange(self.order+1):
      n = self.order+1 - m
      tmp[m] = np.sum([self.B[m,i]*y**i for i in np.arange(n)],axis=0)
    for m in np.arange(self.order+1):
      dy += tmp[m]*x**m

    return np.vstack((x+dx,y+dy))


@dataclass
class SipMod(Sip):
  ''' Modified SIP convention with the distortion center.

  Attributes:
    order (int):
        The polynomial order of the SIP convention.
    center (ndarray):
        The distortion center.
    A (ndarray):
        The SIP coefficient matrix for the x-coordinate.
        The shape of the matrix should be (order+1, order+1).
    B (ndarray):
        The SIP coefficient matrix for the y-coordinate.
        The shape of the matrix should be (order+1, order+1).
  '''
  center: np.ndarray

  def __post_init__(self):
    assert self.center.size == 2, \
      'The center position should have two elements.'
    self.center = self.center.flatten()


class __BaseSipDistortion(object):
  def __call__(self, position: np.ndarray):
    ''' Distortion Function with SIP convention

    This function converts _correct_ coordinates into _distorted_ coordinates.
    The distorted coordinates are obtained by an interative method.

    Parameters:
      position (ndarray):
          A numpy.array with the shape of (2, Nsrc). The first element
          contains the x-positions, while the second element contains
          the y-positions.

    Return:
      A numpy.ndarray of the input coordinates.
    '''
    position = np.array(position).reshape((2,-1))
    p0,x0,d = position, position.copy(),np.infty
    for n in range(100):
      x1 = x0 + (p0-self.apply(x0))
      f,d,x0 = d,np.square(x1-x0).mean(),x1
      if d < 1e-24: break
      if f/d < 1.0: break
    else:
      raise RuntimeError('Iteration not converged.')
    return x0

  def __solve__(self, position: np.ndarray):
    ''' Distortion Function with SIP convention

    This function converts _correct_ coordinates into _distorted_ coordinates.
    The distorted coordinates are obtained by a least square minimization.
    Note that this method fails if the number of positions is too large.

    Parameters:
      position (ndarray):
          A numpy.array with the shape of (2, Nsrc). The first element
          contains the x-positions, while the second element contains
          the y-positions.

    Return:
      A numpy.ndarray of the input coordinates.
    '''
    p0 = np.array(position).flatten()
    func = lambda x: p0-self.apply(x.reshape((2,-1))).flatten()
    result = least_squares(func, p0,
        loss='linear', ftol=1e-15, xtol=1e-15, gtol=1e-15)
    assert result.success is True, \
      'failed to perform a SIP inverse conversion.'
    return result.x.reshape((2,-1))


class SipDistortion(Sip,__BaseSipDistortion):
  pass


class SipModDistortion(SipMod,__BaseSipDistortion):
  pass
