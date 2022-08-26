#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define distortion function by the SIP convention '''

from dataclasses import dataclass, field
from .base import BaseDistortion
import numpy as np


@dataclass
class Sip:
    ''' SIP (simple imaging polynomical) convention

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
        self.center = np.array((0, 0)).reshape((2, 1))
        dim = self.order + 1
        assert self.order >= 0, \
            'The polynomical order should be non-negative.'
        assert self.A.shape == (dim, dim), \
            f'The shape of A matris should be ({dim}, {dim}).'
        assert self.B.shape == (dim, dim), \
            f'The shape of B matris should be ({dim}, {dim}).'

    def normalize(self, position: np.ndarray):
        ''' Normalize position '''
        position = position.copy()
        return position - self.center

    def apply(self, position: np.ndarray):
        ''' Modify xy-coordinates with the SIP function

        Arguments:
          position (ndarray):
              An array contains the list of coordinates. The shape of the array
              should be (2, Nsrc), where Nsrc is the number of sources.

        Returns:
          An ndarray instance contains modified coordinates.
        '''
        N = position.shape[1]
        x, y = self.normalize(position)

        dx = np.zeros_like(x)
        tmp = np.zeros((self.order + 1, N))
        for m in np.arange(self.order + 1):
            n = self.order + 1 - m
            tmp[m] = np.sum([self.A[m, i] * y**i for i in np.arange(n)],
                            axis=0)
        for m in np.arange(self.order + 1):
            dx += tmp[m] * x**m

        dy = np.zeros_like(x)
        tmp = np.zeros((self.order + 1, N))
        for m in np.arange(self.order + 1):
            n = self.order + 1 - m
            tmp[m] = np.sum([self.B[m, i] * y**i for i in np.arange(n)],
                            axis=0)
        for m in np.arange(self.order + 1):
            dy += tmp[m] * x**m

        return position + np.stack((dx, dy))


@dataclass
class AltSip(Sip):
    ''' SIP convention with the displaed distortion center

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
        self.center = np.array(self.center).reshape((2, 1))


class SipDistortion(Sip, BaseDistortion):
    ''' Distortion function with the SIP convention

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
    pass


class AltSipDistortion(AltSip, BaseDistortion):
    ''' Distortion function with the SIP convention

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
    pass
