#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from functools import reduce
from operator import add
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

  def distortion(position):
    ''' Generated distortion function.

    Parameters:
      position: A numpy.array with the shape of (2, Nsrc). The first element
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
          should be (N, 2), where N is the number of sources.

    Return:
      An ndarray instance contains modified coordinates.
    '''
    N = position.shape[0]
    cx,cy = self.center
    x = position[:,0] - cx
    y = position[:,1] - cy

    dx = np.zeros_like(x)
    tmp = np.zeros((N,self.order+1))
    for m in np.arange(self.order+1):
      n = self.order+1 - m
      tmp[:,m] = np.sum([self.A[m,i]*y**i for i in np.arange(n)],axis=0)
    for m in np.arange(self.order+1):
      dx += tmp[:,m]*x**m

    dy = np.zeros_like(x)
    tmp = np.zeros((N,self.order+1))
    for m in np.arange(self.order+1):
      n = self.order+1 - m
      tmp[:,m] = np.sum([self.B[m,i]*y**i for i in np.arange(n)],axis=0)
    for m in np.arange(self.order+1):
      dy += tmp[:,m]*x**m

    return np.vstack((x+dx+cx,y+dy+cy)).T


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
    assert self.center.size == 2
    self.center = self.center.flatten()
