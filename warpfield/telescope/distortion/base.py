#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define distortion base classes '''

import numpy as np


class BaseDistortion:

    def __call__(self, position: np.ndarray):
        ''' Distortion function with SIP convention

        Converts _correct_ coordinates into _distorted_ coordinates.
        The distorted coordinates are obtained by an interative method.

        Arguments:
          position (ndarray):
              A numpy.array with the shape of (2, Nsrc). The first element
              contains the x-positions, while the second element contains
              the y-positions.

        Returns:
          A numpy.ndarray of the input coordinates.
        '''
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
