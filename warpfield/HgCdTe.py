#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, List
from astropy.coordinates import SkyCoord, Angle
from astropy.units.quantity import Quantity
from shapely.geometry import Polygon
import astropy.units as u

from .telescope import Optics,Detector,Telescope
from .telescope import identity_transformation

side = (4096*10*u.um/2).to_value(u.um)
square = Polygon(([-side,-side],[side,-side],[side,side],[-side,side]))

def get_jasmine(
    pointing: SkyCoord,
    position_angle: Angle,
    distortion: Callable = identity_transformation):
  optics = Optics(
    pointing,
    position_angle,
    focal_length = 3.86*u.m,
    diameter     = 0.3*u.m,
    valid_region = square,
    margin       = 5000*u.um,
    distortion   = distortion)
  detectors = [
    Detector(4096, 4096, pixel_scale = 10*u.um),
  ]

  return Telescope(optics=optics, detectors=detectors)
