#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define the nominal JASMINE telescope design '''

from typing import Callable
from astropy.coordinates import SkyCoord, Angle
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import astropy.units as u
import numpy as np

from .optics import Optics
from .detector import Detector
from .telescope import Telescope
from .distortion import identity_transformation


def square_mask(
        naxis = 1920,
        pixel_scale = 10 * u.um,
        sensor_gap = 3.0 * u.mm):
    ''' Generate a square field of view '''
    side = (naxis * pixel_scale + sensor_gap / 2.0).to_value(u.um)
    return Polygon([
        [-side, -side],
        [+side, -side],
        [+side, +side],
        [-side, +side],
    ])


def octagonal_mask():
    ''' Generate an octagonal field of view '''
    c45 = np.cos(np.pi / 4)
    square = square_mask()
    affine_matrix = 1.0225 * np.array([c45, -c45, c45, c45, 0, 0])
    return square.intersection(affine_transform(square, affine_matrix))


def get_jasmine(
      pointing: SkyCoord,
      position_angle: Angle,
      distortion: Callable = identity_transformation,
      octagonal: bool = False):
    ''' Generate JASMINE telescope

    Arguments:
      pointing (SkyCoord):
          A direction of the telescope pointing.
      position_angle (Angle):
          A position angle of the telescope.
      distortion (function):
          A distortion function.
          `identity_transformation` is set if not specified.
      octagonal (bool):
          Set to True if the field of view is octagonal.

    Returns:
      A telescope instance defined by the nominal JASMINE design.
    '''
    optics = Optics(
      pointing,
      position_angle,
      focal_length  = 4.86 * u.m,
      diameter      = 0.4 * u.m,
      field_of_view = octagonal_mask() if octagonal else square_mask(),
      margin        = 5000 * u.um,
      distortion    = distortion)

    step = 1920 * 5 * u.um + 1.5 * u.mm
    xx = step * np.array([-1, 1, 1, -1])
    yy = step * np.array([-1, -1, 1, 1])
    pa = Angle([0, 90, 180, 270], unit='degree')
    detectors = [
        Detector(
            naxis1=1920,
            naxis2=1920,
            pixel_scale=10 * u.um,
            offset_dx=x,
            offset_dy=y,
            position_angle=p)
        for x, y, p in zip(xx, yy, pa)
    ]

    return Telescope(optics=optics, detectors=detectors)
