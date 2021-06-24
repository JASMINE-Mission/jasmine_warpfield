#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Callable, List
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
from astropy.time import Time
from astropy.wcs import WCS
from scipy.spatial.transform import Rotation
import astropy.units as u
import numpy as np

from .util import get_projection


def identity_transformation(position):
  ''' An identity transformation function.

  Parameters:
    position: A tuple of two arrays. The first element contains the
              x-positions, while the second element contains the y-positions.

  Return:
    A numpy.ndarray of the input coordinates.
  '''
  return np.array(position)


@dataclass
class Optics(object):
  ''' Definition of optical components.

  Attributes:
    pointing (SkyCoord)   : the latitude of the telescope pointing.
    pa (Angle)            : the position angle of the telescope.
    focal_length (float)  : the focal length of the telescope in meter.
    diameter (float)      : the diameter of the telescope in meter.
    distortion (function) : a function to distort the focal plane image.
  '''
  pointing: SkyCoord
  pa: Angle            = Angle(0.0, unit='degree')
  focal_length: float  = 7.3
  diameter: float      = 0.4
  distortion: Callable = identity_transformation

  @property
  def scale(self):
    ''' A conversion factor from sky to focal plane. '''
    return (1.0e-6/self.focal_length)/np.pi*180.0*3600

  @property
  def center(self):
    return SkyCoord(0*u.deg,0*u.deg,frame='icrs')

  @property
  def rotation(self):
    ''' Angle set to derive an intermediate world coordinate. '''
    ## use the ICRS frame in calculation.
    icrs = self.pointing.icrs
    ## calculate position angle in the ICRS frame.
    north = self.pointing.directional_offset_by(0.0,1*u.arcsec)
    delta = self.pointing.icrs.position_angle(north)
    return np.array((icrs.ra.rad,-icrs.dec.rad,self.pa.rad-delta.rad))

  def set_distortion(self, distortion):
    ''' Assign distortion function.

    The argument of the distortion function should be a

    Parameters:
      distortion (function): a function to distort focal plane image.
    '''
    self.distortion = distortion

  def imaging(self, sources, epoch=Time.now()):
    ''' Map celestial positions onto the focal plane.

    Parameters:
      epoch (Time): the epoch of the observation.

    Return:
      A numpy.ndarray with the array shape of (2, N(sources)).
      The first array contains the x-coordinates and the second array
      contains the y-coordinates on the focal plane in units of micron.
    '''
    try:
      obj = sources.apply_space_motion(epoch)
    except Exception as e:
      print(str(e))
      obj = sources
    xyz = obj.transform_to('icrs').cartesian.xyz
    r = Rotation.from_euler('zyx', -self.rotation)
    pqr = r.as_matrix() @ xyz
    obj = SkyCoord(pqr.T, obstime=epoch,
            representation_type='cartesian').transform_to('icrs')
    proj = get_projection(self.center,self.scale)
    return self.distortion(obj.to_pixel(proj, origin=0))


@dataclass
class DetectorOffset(object):
  ''' Offset of the detector from the focal plane origin.

  Attributes:
    dx (float): the offset along with the x-axis in micron.
    dy (float): the offste along with the y-axis in micron.
  '''
  dx: float = 0
  dy: float = 0


@dataclass
class PixelDisplacement(object):
  ''' Definition of the pixel non-uniformity.

  Attributes:
    dx (ndarray): a two dimensional array with the same size of the detector.
                  each element contains the x-displacement of the pixel.
    dy (ndarray): a two dimensional array with the same size of the detector.
                  each element contains the y-displacement of the pixel.
  '''
  dx: np.ndarray = None
  dy: np.ndarray = None


  def initialize(self, naxis1, naxis2):
    ''' Initialize the displacement array with zeros.

    Parameters:
      naxis1 (int): the detector size along with NAXIS1.
      naxis2 (int): the detector size along with NAXIS2.
    '''
    self.dx = np.zeros((naxis2, naxis1))
    self.dy = np.zeros((naxis2, naxis1))


  def evaluate(self, position):
    ''' Evaluate the source position displacement.

    Parameters:
      position (ndarray): a numpy.ndarray with the shape of (2, N(sources)).
                          the first array contains the x-coordinates, while
                          the second does the y-coordinates.
    Note:
      Not implemented yet.
    '''
    return position


@dataclass
class Detector(object):
  ''' Definition of a detector.
  '''
  naxis1: int = 4096
  naxis2: int = 4096
  pixel_scale: float = 15
  offset: DetectorOffset = None
  displacement: PixelDisplacement = None

  @property
  def width(self):
    return self.naxis1*self.pixel_scale
  @property
  def height(self):
    return self.naxis2*self.pixel_scale
  @property
  def xrange(self):
    return self.offset.dx + np.array((-self.width/2,self.width/2))
  @property
  def yrange(self):
    return self.offset.dy + np.array((-self.height/2,self.height/2))

  def __post_init__(self):
    if self.offset is None:
      self.offset = DetectorOffset(0,0)
    if self.displacement is None:
      self.displacement = PixelDisplacement()
      self.displacement.initialize(self.naxis1,self.naxis2)

  def capture(self, position):
    '''
    '''
    x = position[0]
    y = position[1]
    xf = ((self.xrange[0] < x) & (x < self.xrange[1]))
    yf = ((self.yrange[0] < y) & (y < self.yrange[1]))
    return self.displacement.evaluate(position[:,xf&yf])


@dataclass
class Telescope(object):
  '''
  '''
  pointing: SkyCoord
  pa: Angle
  optics: Optics     = field(init=False)
  detectors: List[Detector] = None

  def __post_init__(self):
    self.optics = Optics(self.pointing, self.pa)
    if self.detectors is None:
      self.detectors = [Detector(),]

  def set_distortion(self, distortion):
    ''' Set the optics distortion function.

    Parameters:
      distortion (function): a function to distort focal plane image.
    '''
    self.optics.set_distortion(distortion)

  def observe(self, sources, epoch=Time.now()):
    ''' Observe astronomical sources.

    Parameters:
      sources (SkyCoord): a list of astronomical sources.
      epoch (Time): the datetime of the observation.

    Return:
      A numpy.ndarray with the shape of (N(detector), 2, N(source)).
      The first index specifies the detector of the telescope.
      A two dimensional array is assigned for each detector. The first
      line is the coordinates along the NAXIS1 axis, and the second one
      is the coordinates along the NAXIS2 axis.
    '''
    position = self.optics.imaging(sources, epoch)
    ret = []
    for det in self.detectors:
      ret.append(det.capture(position))

    return np.array(ret)
