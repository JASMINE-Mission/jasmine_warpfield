#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
