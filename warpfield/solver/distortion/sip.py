#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' distortion function defined by the SIP convention '''

from jax.lax import scan
import jax.numpy as jnp


def maxord(P):
  ''' calculate the maximum order from the coefficients

  Args:
    P: coefficients of a polynomial expansion.

  Retruns:
    the maximum order of the polynomical expansion defined by the given
    polyomial coefficients.
  '''
  return int((jnp.sqrt(1 + 8 * P.size + 24) - 1) / 2)


def split(P):
  ''' split the coefficients by the polynomial order

  Args:
    P: coefficients of a polynomial expansion.

  Returns:
    polynomial expansion coefficients grouped by the order.
  '''
  idx = jnp.cumsum(jnp.arange(3, maxord(P)))
  return jnp.split(P, idx)


def polymap(P, xy):
  ''' calculate a two-dimensional polynomical expansion

  Args:
    P: coefficients of a polynomial expansion.
    xy: original coordinates on the focal plane.

  Returns:
    (N,2) list of converted coordinates.
  '''
  def inner(M, p):
    ''' inner function to calculate a polynomical expansion

    Args:
      M: (m,n) integer power index pair.
      p: scale coefficient.

    Returns:
      calclated cordinates (p * x**m * y**n).
    '''
    m, n = M
    return [m - 1, n + 1], p * xy[:, 0]**m * xy[:, 1]**n

  _, pq = scan(inner, [len(P) - 1, 0], P)
  return pq.sum(axis=0)


def distortion(A, B, xy):
  x, y = xy[:, 0], xy[:, 1]
  dx = polymap(A[0:3], xy) + polymap(A[3:7], xy) \
    + polymap(A[7:12], xy) + polymap(A[12:18], xy)
  dy = polymap(B[0:3], xy) + polymap(B[3:7], xy) \
    + polymap(B[7:12], xy) + polymap(B[12:18], xy)
  return xy + jnp.stack([dx, dy]).T
