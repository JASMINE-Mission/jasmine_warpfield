#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from jax.config import config
from numpyro import set_platform
config.update('jax_enable_x64', True)

from jax.lax import scan
import jax.numpy as jnp
import numpy as np

from numpyro.infer import NUTS, MCMC
import numpyro


def maxdim(P):
  return int((np.sqrt(1+8*P.size+24)-1)/2)


def split(P):
  idx = jnp.cumsum(jnp.arange(3,maxdim(P)))
  return jnp.split(P, idx)


def polymap(P, xy):
  def inner(M, p):
    m,n = M
    return [m-1,n+1], p*xy[:,0]**m*xy[:,1]**n
  _,pq = scan(inner, [len(P)-1, 0], P)
  return pq.sum(axis=0)


def distortion(A, B, xy):
  x,y = xy[:,0],xy[:,1]
  dx = polymap(A[0:3], xy) + polymap(A[3:7], xy) \
    + polymap(A[7:12], xy) + polymap(A[12:18], xy)
  dy = polymap(B[0:3], xy) + polymap(B[3:7], xy) \
    + polymap(B[7:12], xy) + polymap(B[12:18], xy)
  return xy+jnp.stack([dx,dy]).T


def distortion20(A, B, xy):
  x,y = xy[:,0],xy[:,1]
  dx = polymap(A[0:2], xy) + polymap(A[2:5], xy) + polymap(A[5:9], xy) \
    + polymap(A[9:14], xy) + polymap(A[14:20], xy)
  dy = polymap(B[0:2], xy) + polymap(B[2:5], xy) + polymap(B[5:9], xy) \
    + polymap(B[9:14], xy) + polymap(B[14:20], xy)
  return xy+jnp.stack([dx,dy]).T
