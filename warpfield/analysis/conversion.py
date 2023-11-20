#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp


def degree_to_radian(theta):
    ''' Convert degree to radian '''
    return theta * jnp.pi / 180.


def rotation_matrix(theta):
    ''' Calculate rotation matrix R '''
    rot = [jnp.cos(theta), -jnp.sin(theta), jnp.sin(theta), jnp.cos(theta)]
    return jnp.array(rot).reshape([2, 2])
