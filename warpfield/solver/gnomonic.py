#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as jnp
import numpy as np

from jax import vmap
from numpyro.infer import NUTS, MCMC
import numpyro


def proj(a0, d0, t0, a, d, s):
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

projection = vmap(proj, (0,0,0,0,0,0),0)