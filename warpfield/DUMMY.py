#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, List
from astropy.coordinates import SkyCoord, Angle
from astropy.units.quantity import Quantity
import astropy.units as u

from .telescope import Optics,Detector,Telescope
from .telescope import identity_transformation

def get_jasmine(
    pointing: SkyCoord,
    position_angle: Angle,
    distortion: Callable = identity_transformation):
  optics = Optics(
    pointing,
    position_angle,
    focal_length = 7.3*u.m,
    diameter     = 0.4*u.m,
    fov_radius   = 30000*u.um,
    distortion   = distortion)
  detectors = [
    Detector(4096, 4096, pixel_scale = 10*u.um),
  ]

  return Telescope(optics=optics, detectors=detectors)
