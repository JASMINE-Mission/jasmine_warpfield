#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Distortion function defined by the SIP convention '''

from jax.lax import scan
from jax import Array
import jax.numpy as jnp
import numpy as np

from .base import Distortion


__all__ = ['SIPDistortion']


def _polymap(coeff, xy):
    ''' Calculate a two-dimensional polynomical expansion

    Arguments:
        coeff: Coefficients of a polynomial expansion.
        xy: Original coordinates on the focal plane.

    Returns:
        A (N,2) list of converted coordinates.
    '''

    def inner(order, coeff):
        ''' Inner function to calculate a polynomical expansion

        Arguments:
            order: A (m,n) integer power index pair.
            coeff: A scale coefficient.

        Returns:
            A list of calclated cordinates (p * x**m * y**n).
        '''
        m, n = order
        return [m - 1, n + 1], coeff * xy[:, 0]**m * xy[:, 1]**n

    _, pq = scan(inner, [len(coeff) - 1, 0], coeff)
    return pq.sum(axis=0)


def _distortion(sip_a, sip_b, xy):
    ''' Calculate displacements using the SIP coefficients

    The SIP coefficients sip_a and sip_b should contains 18 coefficients.
    The coefficients do not contain the Affine-transformation term.

    - elements 0-2:   second order coefficients
    - elements 3-6:   third order coefficients
    - elements 7-11:  fourth order coefficients
    - elements 12-17: fifth order coefficients

    Arguments:
        sip_a: A list of 5th-order SIP coefficients for x-axis.
        sip_b: A list of 5th-order SIP coefficients for y-axis.
        xy: Original coordinates on the focal plane.

    Returns:
        Coordinate displacements on the focal plane.
    '''
    scale = np.exp(
        -np.log(10) * 4 *
        np.array([2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]))
    sip_a *= scale
    sip_b *= scale
    dx = _polymap(sip_a[0:3], xy) + _polymap(sip_a[3:7], xy) \
        + _polymap(sip_a[7:12], xy) + _polymap(sip_a[12:18], xy)
    dy = _polymap(sip_b[0:3], xy) + _polymap(sip_b[3:7], xy) \
        + _polymap(sip_b[7:12], xy) + _polymap(sip_b[12:18], xy)
    return jnp.stack([dx, dy]).T


class SIPDistortion(Distortion):
    ''' Fifth-order SIP distortion represented as a PyTree

    Attributes:
        coeff_x: Coefficients for x-axis displacements with shape ``(18,)``.
        coeff_y: Coefficients for y-axis displacements with shape ``(18,)``.
    '''

    coeff_x: Array
    coeff_y: Array

    def __init__(self, coeff_x, coeff_y):
        coeff_x = jnp.asarray(coeff_x, dtype=float)
        coeff_y = jnp.asarray(coeff_y, dtype=float)

        if coeff_x.shape != (18,):
            raise ValueError('`coeff_x` should have shape (18,).')
        if coeff_y.shape != (18,):
            raise ValueError('`coeff_y` should have shape (18,).')

        self.coeff_x = coeff_x
        self.coeff_y = coeff_y

    def __call__(self, xy):
        ''' Calculate coordinate displacements on the focal plane '''
        xy = jnp.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_coordinate, 2).')
        return _distortion(self.coeff_x, self.coeff_y, xy)


if __name__ == '__main__':
    from timeit import timeit

    x = jnp.linspace(-1, 1, 201)
    xy = jnp.stack([x, x]).T
    coeff = jnp.array([0.0, 0.0, 0.4])

    print('\nBenchmark of the polynomial map:\n')
    print('  execution time: {:.6f}'.format(
        timeit(lambda: _polymap(coeff, xy), number=25) / 25))

    print('\nBenchmark of the distortion function:\n')
    coeff_a = jnp.zeros(18)
    coeff_b = jnp.zeros(18)
    print('  execution time: {:.6f}'.format(
        timeit(lambda: _distortion(coeff_a, coeff_b, xy), number=100)))
