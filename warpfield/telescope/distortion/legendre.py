#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define distortion function by the Legendre polynomials '''

from dataclasses import dataclass
from .base import BiPolynomialFunction, InvertibleFunction
from numpy.polynomial.legendre import legval2d
import numpy as np


class Legendre(BiPolynomialFunction, InvertibleFunction):
    ''' Displacement function by the Legendre polynomials

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
        return (position.copy() - self.get_center()) / self.scale

    def apply(self, position: np.ndarray):
        ''' Modify xy-coordinates with the Legendre polynomial function

        Arguments:
          position (ndarray):
              An array contains the list of coordinates.
              The shape of the array should be (2, Nsrc), where Nsrc is
              the number of sources.

        Returns:
          An ndarray instance contains modified coordinates.
        '''
        x, y = self.normalize(position)

        dx = legval2d(x, y, self.A)
        dy = legval2d(x, y, self.B)

        return position + np.stack((dx, dy))


@dataclass
class LegendreDistortion(Legendre):
    ''' Distortion function with the Legendre polynomials

    Attributes:
      order (int):
          The maximum order of the Legendre polynomials.
      A (ndarray):
          The coefficient matrix for the x-coordinate.
          The shape of the matrix should be (order+1, order+1).
      B (ndarray):
          The coefficient matrix for the y-coordinate.
          The shape of the matrix should be (order+1, order+1).
    '''
    order: int
    A: np.ndarray
    B: np.ndarray
    scale: float = 30000.


@dataclass
class DisplacedLegendreDistortion(Legendre):
    ''' Displacement function by the Legendre polynomials

    Attributes:
      order (int):
          The maximum order of the Legendre polynomials.
      center (ndarray):
          The distortion center.
      A (ndarray):
          The coefficient matrix for the x-coordinate.
          The shape of the matrix should be (order+1, order+1).
      B (ndarray):
          The coefficient matrix for the y-coordinate.
          The shape of the matrix should be (order+1, order+1).
    '''
    order: int
    center: np.ndarray
    A: np.ndarray
    B: np.ndarray
    scale: float = 30000.
