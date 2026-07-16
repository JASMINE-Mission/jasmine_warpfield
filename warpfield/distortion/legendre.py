#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Distortion function using the Legendre polynomials '''

import equinox as eqx
from jax import Array
from jax.lax import scan
import jax.numpy as jnp
import numpy as np

from .base import Distortion


__all__ = ['LegendreDistortion']


def _val2d(func, x, y, c):
    ''' A helper function to evaluate a 2d-polynomial function

    Arguments:
        func (function): A function to evaluate a polynomial expression.
        x (array): A list of first coordiantes.
        y (array): A list of second coordinates.
        c (array): A list of coefficients.

    Returns:
        An array of elements evalulated at (x, y).
    '''
    assert x.shape == y.shape, \
        'arrays `x` and `y` should have the same shapes.'
    return func(y, func(x, c), tensor=False)


def _legval(x, c, tensor=True):
    ''' Evaluate a one-dimensional Legendre polynomial expansion

    Arguments:
        x (array): A list of evaluation coordinates.
        c (array): A list of coefficients.

    Returns:
        An evaluation of Legendre polynomial expansion.
    '''
    if isinstance(x, jnp.ndarray) and tensor:
        c = c.reshape(c.shape + (1, ) * x.ndim)
    if len(c) <= 2:
        z = jnp.zeros(list([2, *c.shape[1:]]))
        c = jnp.concatenate([c, z], axis=0)

    # Legendre polynomial is calculated by the following iterations:
    #
    # nd = len(c)
    # c0 = c[-2]
    # c1 = c[-1]
    # for i in range(3, len(c)+1):
    #     tmp = c0
    #     nd = nd - 1
    #     c0 = c[-i] - (c1*(nd - 1))/nd
    #     c1 = tmp + (c1*x*(2*nd - 1))/nd

    init = [
        len(c),
        c[-2] * jnp.ones_like(x),
        c[-1] * jnp.ones_like(x),
    ]

    def evaluate(init, coeff):
        nd, c0, c1 = init
        c0 = coeff - (c1 * (nd - 2)) / (nd - 1)
        c1 = init[1] + (c1 * x * (2 * nd - 3)) / (nd - 1)
        return [nd - 1, c0, c1], None

    _, null = scan(evaluate, init, c[:-2][::-1])
    nd, c0, c1 = _

    return c0 + c1 * x


def _legval2d(x, y, c):
    ''' Evaluate a two-dimensional Legendre polynomial expansion

    Arguments:
        x (array): A list of evaluation coordinates.
        y (array): A list of evaluation coordinates.
        c (array): A list of coefficients.

    Returns:
        An evaluation of Legendre polynomial expansion.
    '''
    c = jnp.atleast_2d(c)
    return _val2d(_legval, x, y, c)


def _map_coeff_5th(c):
    ''' Convert 18-element coefficient array into a 6x6 matrix '''

    # coeff :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
    # matrix: 12  7  2 18 18  8  3 24 19 14  9  4 30 25 20 15 10  5

    #  0  1  2  3  4  5
    #  6  7  8  9 10 11
    # 12 13 14 15 16 17
    # 18 19 20 21 22 23
    # 24 25 26 27 28 29
    # 30 31 32 33 34 35

    z0 = (c[0] + c[2]) / 2 - c[9] / 4 - (c[7] + c[11]) * 3 / 8

    return jnp.array([
           z0,     0, c[ 2], c[ 6], c[11], c[17],
            0, c[ 1], c[ 5], c[10], c[16],     0,
        c[ 0], c[ 4], c[ 9], c[15],     0,     0,
        c[ 3], c[ 8], c[14],     0,     0,     0,
        c[ 7], c[13],     0,     0,     0,     0,
        c[12],     0,     0,     0,     0,     0,
    ]).reshape((6, 6))


def _distortion(coeff_a, coeff_b, xy):
    ''' Calculate displacements using the Legendre coefficients

    The coefficients coeff_a and coeff_b should contain 18 coefficients.
    The coefficients do not contain the Affine-transformation term.

    - elements 0-2:   second order coefficients
    - elements 3-6:   third order coefficients
    - elements 7-11:  fourth order coefficients
    - elements 12-17: fifth order coefficients

    Arguments:
        coeff_a: A list of 5th-order coefficients for x-axis.
        coeff_b: A list of 5th-order coefficients for y-axis.
        xy: Original coordinates on the focal plane.

    Returns:
        Coordinate displacements on the focal plane.
    '''
    dx = _legval2d(xy[:, 0], xy[:, 1], _map_coeff_5th(coeff_a))
    dy = _legval2d(xy[:, 0], xy[:, 1], _map_coeff_5th(coeff_b))
    return jnp.stack([dx, dy]).T


class LegendreDistortion(Distortion):
    ''' Fifth-order Legendre distortion represented as a PyTree

    Attributes:
        coeff_x: Coefficients for x-axis displacements with shape ``(18,)``.
        coeff_y: Coefficients for y-axis displacements with shape ``(18,)``.
        plane_scale: Scale used to normalize focal-plane coordinates.
    '''

    coeff_x: Array
    coeff_y: Array
    plane_scale: float = eqx.field(static=True)

    def __init__(self, coeff_x, coeff_y, plane_scale):
        coeff_x = jnp.asarray(coeff_x, dtype=float)
        coeff_y = jnp.asarray(coeff_y, dtype=float)
        plane_scale = float(plane_scale)

        if coeff_x.shape != (18,):
            raise ValueError('`coeff_x` should have shape (18,).')
        if coeff_y.shape != (18,):
            raise ValueError('`coeff_y` should have shape (18,).')
        if not np.isfinite(plane_scale) or plane_scale <= 0:
            raise ValueError('`plane_scale` should be finite and positive.')

        self.coeff_x = coeff_x
        self.coeff_y = coeff_y
        self.plane_scale = plane_scale

    def __call__(self, xy):
        ''' Calculate coordinate displacements on the focal plane '''
        xy = jnp.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_coordinate, 2).')
        return _distortion(
            self.coeff_x,
            self.coeff_y,
            xy / self.plane_scale,
        )


if __name__ == '__main__':
    from timeit import timeit

    x = jnp.linspace(-1, 1, 201)
    coeff = np.random.normal(size=(16))

    print('\nBenchmark of 1D-Legendre polynomial:\n')
    print('  execution time: {:.6f}'.format(
        timeit(lambda: _legval(x, coeff), number=25) / 25))

    coeff = np.random.normal(size=(8, 8))

    print('\nBenchmark of 2D-Legendre polynomial:\n')
    print('  execution time: {:.6f}'.format(
        timeit(lambda: _legval2d(x, x, coeff), number=25) / 25))

    xy = jnp.stack([x, x]).T

    print('\nBenchmark of the distortion function:\n')
    coeff_a = jnp.array([0.1] + [0.0] * 17)
    coeff_b = jnp.array([0.0, 0.1] + [0.0] * 16)
    print('  execution time: {:.6f}'.format(
        timeit(lambda: _distortion(coeff_a, coeff_b, xy), number=100)))

    print('\nDistortion value shold be zero at the origin:\n')
    xy = jnp.array([[0.0, 0.0]])
    for n in range(10):
        coeff_a = jnp.array(np.random.normal(size=(18)))
        coeff_b = jnp.array(np.random.normal(size=(18)))
        print('  (case {0}): [{1:+.2e} {2:+.2e}]'.format(
            n, *_distortion(coeff_a, coeff_b, xy)[0]))
