#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Distortion function using the Legendre polynomials """

from jax.lax import scan
from jax import jit
import jax.numpy as jnp


def val2d(func, c, x, y):
    """ A helper function to evaluate a 2d-polynomial function

    Arguments:
      func (function): A function to evaluate a polynomial expression.
      c (array): A list of coefficients.
      x (array): A list of first coordiantes.
      y (array): A list of second coordinates.

    Returns:
      An array of elements evalulated at (x, y).
    """
    assert x.shape == y.shape, \
      'arrays `x` and `y` should have the same shapes.'
    return func(y, func(x, c), tensor=False)


def legval(x, c, tensor=True):
    """ Evaluate a one-dimensional Legendre polynomial expansion

    Arguments:
      x (array): A list of evaluation coordinates.
      c (array): A list of coefficients.

    Returns:
      An evaluation of Legendre polynomial expansion.
    """
    if isinstance(x, jnp.ndarray) and tensor:
        c = c.reshape(c.shape + (1, ) * x.ndim)
    if len(c) <= 2:
        z = jnp.zeros(list([2, *c.shape[1:]]))
        c = jnp.concatenate([c, z], axis=0)

    ## Legendre polynomial is calculated by the following iterations:
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


def legval2d(x, y, c):
    """ Evaluate a two-dimensional Legendre polynomial expansion

    Arguments:
      x (array): A list of evaluation coordinates.
      y (array): A list of evaluation coordinates.
      c (array): A list of coefficients.

    Returns:
      An evaluation of Legendre polynomial expansion.
    """
    c = jnp.atleast_2d(c)
    return val2d(legval, c, x, y)


if __name__ == '__main__':
    from timeit import timeit

    x = jnp.linspace(-1, 1, 11)
    c = jnp.array([0.0, 0.0, 0.0, 0.4])

    print(timeit(lambda: legval(x, c), number=100))

    c = jnp.array([[0.1, 0.1], [0.1, 0.0]])
    legval2d_cc = jit(legval2d)

    print(timeit(lambda: legval2d(x, x, c), number=100))
    print(timeit(lambda: legval2d_cc(x, x, c), number=100))
