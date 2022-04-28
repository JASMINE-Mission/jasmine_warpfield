#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Define distortion function by the Legendre polynomials """

from dataclasses import dataclass, field
from .base import BaseDistortion
from numpy.polynomial.legendre import legval2d
import numpy as np


@dataclass
class Legendre:
    """ Displacement function by the Legendre polynomials

    Attributes:
      order (int):
          The polynomial order of the Legendre convention.
      A (ndarray):
          The coefficient matrix for the x-coordinate.
          The shape of the matrix should be (order+1, order+1).
      B (ndarray):
          The coefficient matrix for the y-coordinate.
          The shape of the matrix should be (order+1, order+1).
    """
    order: int
    center: np.ndarray = field(init=False)
    A: np.ndarray
    B: np.ndarray
    scale: float = 30000.

    def __post_init__(self):
        self.center = np.array((0, 0)).reshape((2, 1))
        assert self.order >= 0, \
          f'The polynomical order should be non-negative.'
        assert self.A.shape == (self.order+1,self.order+1), \
          f'The shape of A matris should be ({self.order+1}, {self.order+1}).'
        assert self.B.shape == (self.order+1,self.order+1), \
          f'The shape of B matris should be ({self.order+1}, {self.order+1}).'

    def normalize(self, position: np.ndarray):
        """ Normalize position """
        return (position.copy() - self.center) / self.scale

    def apply(self, position: np.ndarray):
        """ Modify xy-coordinates with the Legendre polynomial function

        Arguments:
          position (ndarray):
              An array contains the list of coordinates.
              The shape of the array should be (2, Nsrc), where Nsrc is the number of sources.

        Returns:
          An ndarray instance contains modified coordinates.
        """
        x, y = self.normalize(position)

        dx = legval2d(x, y, self.A)
        dy = legval2d(x, y, self.B)

        return position + np.stack((dx, dy))


@dataclass
class AltLegendre(Legendre):
    """ Displacement function by the Legendre polynomials

    Attributes:
      order (int):
          The polynomial order of the Legendre convention.
      center (ndarray):
          The distortion center.
      A (ndarray):
          The coefficient matrix for the x-coordinate.
          The shape of the matrix should be (order+1, order+1).
      B (ndarray):
          The coefficient matrix for the y-coordinate.
          The shape of the matrix should be (order+1, order+1).
    """
    center: np.ndarray

    def __post_init__(self):
        assert self.center.size == 2, \
          'The center position should have two elements.'
        self.center = self.center.reshape((2, 1))


class LegendreDistortion(Legendre, BaseDistortion):
    """ distortion function with the Legendre (simple imaging polynomical) convention

    Attributes:
      order (int):
          The polynomial order of the Legendre convention.
      A (ndarray):
          The Legendre coefficient matrix for the x-coordinate.
          The shape of the matrix should be (order+1, order+1).
      B (ndarray):
          The Legendre coefficient matrix for the y-coordinate.
          The shape of the matrix should be (order+1, order+1).
    """
    pass


class AltLegendreDistortion(AltLegendre, BaseDistortion):
    """ Distortion function with the displaced Legendre polynomials

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
    """
    pass
