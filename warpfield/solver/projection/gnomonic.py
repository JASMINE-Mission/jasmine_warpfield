#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' gnomonic projection '''

import jax.numpy as jnp
from jax import vmap


def projection(a0, d0, t0, a, d, s):
  ''' gnomonic projection of the spherical coordinates.

  Args:
    a0: right ascension of the telescope center.
    d0: declinatoin of the telescope center.
    t0: position angle of the telescope.
    a: right ascension of the target.
    d: declination of the target.
    s: physical scale of the focal plane (mm/deg).

  Returns:
    converted coordinates on the focal plane
  '''
  d2r = jnp.pi/180.0
  a0,d0,a,d,t0 = a0*d2r,d0*d2r,a*d2r,d*d2r,t0*d2r
  A = jnp.array([
    jnp.cos(t0),-jnp.sin(t0),jnp.sin(t0),jnp.cos(t0)]).reshape([2,2])
  x = jnp.sin(d0)*jnp.cos(d)*jnp.cos(a-a0)-jnp.cos(d0)*jnp.sin(d)
  y = jnp.cos(d)*jnp.sin(a-a0)
  z = jnp.cos(d0)*jnp.cos(d)*jnp.cos(a-a0)+jnp.sin(d0)*jnp.sin(d)
  R   = 180.0/jnp.pi*jnp.sqrt(1/z**2-1)
  phi = jnp.arctan2(x,-y)
  X,Y =  R*jnp.cos(phi),-R*jnp.sin(phi)
  return (A@jnp.stack([X,Y])).T*s

gnomonic = vmap(projection, (0,0,0,0,0,0),0)
