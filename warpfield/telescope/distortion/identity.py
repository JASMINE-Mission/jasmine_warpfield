#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" define identity transformation for distortion """

import numpy as np


def identity_transformation(position):
    """ An identity transformation function.

    This function is an fallback function for the image distortion.
    The function requires a tuple of two arrays. The first and second elements
    are the x- and y-positions on the focal plane without any distortion,
    respectively. This function returns the positions as they are.

    Arguments:
      position:
          A numpy.array with the shape of (2, Nsrc). The first element contains
          the x-positions, while the second element contains the y-positions.

    Returns:
      A numpy.ndarray of the input coordinates.
    """
    return np.array(position)
