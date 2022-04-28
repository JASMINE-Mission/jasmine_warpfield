#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Define distortion base classes """

from scipy.optimize import least_squares
import numpy as np


class BaseDistortion:

    def __call__(self, position: np.ndarray):
        """ Distortion function with SIP convention

        This function converts _correct_ coordinates into _distorted_ coordinates.
        The distorted coordinates are obtained by an interative method.

        Arguments:
          position (ndarray):
              A numpy.array with the shape of (2, Nsrc). The first element
              contains the x-positions, while the second element contains
              the y-positions.

        Returns:
          A numpy.ndarray of the input coordinates.
        """
        position = np.array(position).reshape((2, -1))
        p0, x0, d = position, position.copy(), np.infty
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

    def __solve__(self, position: np.ndarray):
        """ Distortion function with SIP convention

        This function converts _correct_ coordinates into _distorted_ coordinates.
        The distorted coordinates are obtained by a least square minimization.
        Note that this method fails if the number of positions is too large.

        Arguments:
          position (ndarray):
              A numpy.array with the shape of (2, Nsrc). The first element
              contains the x-positions, while the second element contains
              the y-positions.

        Returns:
          A numpy.ndarray of the input coordinates.
        """
        p0 = np.array(position).flatten()
        func = lambda x: p0 - self.apply(x.reshape((2, -1))).flatten()
        result = least_squares(func,
                               p0,
                               loss='linear',
                               ftol=1e-15,
                               xtol=1e-15,
                               gtol=1e-15)
        assert result.success is True, \
          'failed to perform an inverse conversion.'
        return result.x.reshape((2, -1))
