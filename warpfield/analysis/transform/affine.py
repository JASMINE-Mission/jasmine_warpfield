#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Affine transformation '''

from jax import vmap

from ..conversion import degree_to_radian, rotation_matrix


def __inner_func(xy, rot, offset, scale):
    ''' Convert the focal-plane coordinates into the detector coordinates

    Arguments:
      xy: Focal-plane coordinates (mm).
      rot: A rotation angle in degree.
      offset: Focal-plane offsets (mm).
      scale: Physical pixel sizes (mm).


    Returns:
      Converted coordinates on the detector.
    '''
    rot = degree_to_radian(rot)
    return rotation_matrix(rot) @ ((xy - offset).T) / scale


transform = vmap(__inner_func, (0, 0, 0, 0), 0)
