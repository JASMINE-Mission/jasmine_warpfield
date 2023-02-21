#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define distortion function by the SIP convention '''

from dataclasses import dataclass
from .base import BiPolynomialFunction, InvertibleFunction
import numpy as np


class Sip(BiPolynomialFunction, InvertibleFunction):
    ''' SIP (simple imaging polynomical) convention

    This class does not work properly by itself.
    The following attributes should be defined in child classes.

    Attributes:
      order: the maximum order of the polynomials.
      center: the distortion center (optional).
      A: coefficients for the x-coordinate.
      B: coefficients for the y-coordinate.
    '''

    def normalize(self, position: np.ndarray):
        ''' Normalize position '''
        position = position.copy()
        return position - self.get_center()

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
class SipDistortion(Sip):
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
    order: int
    A: np.ndarray
    B: np.ndarray


@dataclass
class DisplacedSipDistortion(Sip):
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
    order: int
    center: np.ndarray
    A: np.ndarray
    B: np.ndarray
