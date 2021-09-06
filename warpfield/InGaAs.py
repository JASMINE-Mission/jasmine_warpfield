#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, List
from astropy.coordinates import SkyCoord, Angle
from astropy.units.quantity import Quantity
import astropy.units as u
import numpy as np

from .telescope import Optics,Detector,Telescope
from .telescope import identity_transformation

def get_jasmine(
    pointing: SkyCoord,
    position_angle: Angle,
    distortion: Callable = identity_transformation):
  optics = Optics(
    pointing,
    position_angle,
    focal_length = 3.86*u.m,
    diameter     = 0.3*u.m,
    fov_radius   = 30000*u.um,
    distortion   = distortion)

  arr = np.arange(-1,2,)*18000*u.um
  xx,yy = np.meshgrid(arr,arr)
  detectors = [
    Detector(1024, 1024, pixel_scale=15*u.um, offset_dx=x, offset_dy=y)
    for x,y in zip(xx.flat,yy.flat)
  ]

  return Telescope(optics=optics, detectors=detectors)
