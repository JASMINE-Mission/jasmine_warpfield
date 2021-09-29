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
    focal_length = 4.86*u.m,
    diameter     = 0.4*u.m,
    fov_radius   = 30000*u.um,
    distortion   = distortion)

  arr = np.array([-1,1])*(1920*5*u.um+1.5*u.mm)
  xx,yy = np.meshgrid(arr,arr)
  detectors = [
    Detector(1920, 1920, pixel_scale=10*u.um, offset_dx=x, offset_dy=y)
    for x,y in zip(xx.flat,yy.flat)
  ]

  return Telescope(optics=optics, detectors=detectors)
