#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define distortion base classes '''

import numpy as np


class BiPolynomialFunction:
    ''' Requiremnts to define a bi-polynomial function

    This class does not work properly by itself.
    The following attributes should be defined in child classes.

    Attributes:
      order: the maximum order of the polynomials.
      center: the distortion center (optional).
      A: coefficients for the x-coordinate.
      B: coefficients for the y-coordinate.
    '''

    def __post_init__(self):
        ''' This function will be called from a child class '''
        self.check_integrity()

    def get_center(self):
        ''' Return (0,0) if `center` is not defined '''
        if hasattr(self, 'center'):
            return np.reshape(self.center, (2, 1))
        else:
            return np.array((0, 0)).reshape((2, 1))

    def check_integrity(self):
        ''' Check if the attributes are properly defined '''
        dim = self.order + 1
        assert self.order >= 0, \
            'The polynomical order should be non-negative.'
        assert self.get_center().size == 2, \
            'The center position should have two elements.'
        assert self.A.shape == (dim, dim), \
            f'The shape of A matris should be ({dim}, {dim}).'
        assert self.B.shape == (dim, dim), \
            f'The shape of B matris should be ({dim}, {dim}).'


class InvertibleFunction:

    def apply():
        raise NotImplementedError('should be overriden')

    def __call__(self, position: np.ndarray):
        ''' Inverse function of `self.apply()`

        Provides the inverse function of `self.apply()`.
        This function is used to convert _correct_ coordinate into
        _distorted_ coordinates. The distorted coordiantes are obtained
        by an interative method.

        Arguments:
          position (ndarray):
              A numpy.array with the shape of (2, Nsrc). The first element
              contains the x-positions, while the second element contains
              the y-positions.

        Returns:
          A numpy.ndarray of the input coordinates.
        '''
        position = np.array(position).reshape((2, -1))
        p0, x0, d = position, position.copy(), np.inf
        for n in range(100):
            x1 = x0 + (p0 - self.apply(x0))
            f, d, x0 = d, np.square(x1 - x0).mean(), x1
            if d < 1e-24: break
            if abs(1 - f / d) < 1e-3 and d < 1e-16: break
            assert np.isfinite(d), \
                'Floating value overflow detected.'
        else:
            raise RuntimeError(f'Iteration not converged ({d})')
        return x0
