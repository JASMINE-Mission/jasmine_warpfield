#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Callable, List
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
from astropy.time import Time
from astropy.units.quantity import Quantity
from astropy.wcs import WCS
from scipy.spatial.transform import Rotation
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, Point
from shapely.geometry import MultiPoint
from shapely.prepared import prep
from descartes.patch import PolygonPatch
import astropy.units as u
import numpy as np
import pandas as pd
import sys

from .util import get_projection


def identity_transformation(position):
  ''' An identity transformation function.

  This function is an fallback function for the image distortion.
  The function requires a tuple of two arrays. The first and second elements
  are the x- and y-positions on the focal plane without any distortion,
  respectively. This function returns the positions as they are.

  Parameters:
    position: A numpy.array with the shape of (2, Nsrc). The first element
              contains the x-positions, while the second element contains
              the y-positions.

  Return:
    A numpy.ndarray of the input coordinates.
  '''
  return np.array(position)


@dataclass
class Optics(object):
  ''' Definition of optical components.

  Attributes:
    pointing (SkyCoord)    : the latitude of the telescope pointing.
    position_angle (Angle) : the position angle of the telescope.
    focal_length (Quantity): the focal length of the telescope in meter.
    diameter (Quantity)    : the diameter of the telescope in meter.
    valid_region (Polygon) : the valid region of the focal plane.
    distortion (function)  : a function to distort the focal plane image.
  '''
  pointing: SkyCoord
  position_angle: Angle  = Angle(0.0, unit='degree')
  focal_length: Quantity = 7.3*u.m
  diameter: Quantity     = 0.4*u.m
  valid_region: Polygon  = Point(0,0).buffer(30000)
  distortion: Callable   = identity_transformation

  @property
  def scale(self):
    ''' A conversion factor from sky to focal plane in degree/um. '''
    return (1.0*u.rad/self.focal_length).to(u.deg/u.um)

  @property
  def center(self):
    ''' A dummy position to defiine the center of the focal plane. '''
    return SkyCoord(0*u.deg,0*u.deg,frame='icrs')

  @property
  def pointing_angle(self):
    ''' Angle set to define the pointing position and orientation. '''
    ## use the ICRS frame in calculation.
    icrs = self.pointing.icrs
    ## calculate position angle in the ICRS frame.
    north = self.pointing.directional_offset_by(0.0,1*u.arcsec)
    delta = self.pointing.icrs.position_angle(north)
    position_angle = -self.position_angle.rad-delta.rad
    return np.array((icrs.ra.rad,-icrs.dec.rad,position_angle))

  def set_distortion(self, distortion):
    ''' Assign distortion function.

    The argument of the distortion function should be a

    Parameters:
      distortion (function): a function to distort focal plane image.
    '''
    self.distortion = distortion

  def block(self, position):
    ''' Block sources by a certain radius.

    Parameters:
      position (ndarray): source positions on the focal plane w/o distortion.

    Return:
      A boolean array to indicate which sources are inside the field-of-view.
    '''
    mp = MultiPoint(position.T)
    polygon = prep(self.valid_region)
    return np.array([not polygon.contains(p) for p in mp])

  def imaging(self, sources, epoch=None):
    ''' Map celestial positions onto the focal plane.

    Parameters:
      epoch (Time): the epoch of the observation.

    Return:
      A `DataFrame` instance. The DataFrame contains four columns: the "x" and
      "y" columns are the positions on the focal plane in micron, and the "ra"
      and "dec" columns are the original celestial positions in the ICRS frame.
    '''
    try:
      if epoch is not None:
        sources = sources.apply_space_motion(epoch)
    except Exception as e:
      print('No proper motion information is available.', file=sys.stderr)
      print('The positions are not updated to new epoch.', file=sys.stderr)
    icrs = sources.transform_to('icrs')
    xyz = icrs.cartesian.xyz
    r = Rotation.from_euler('zyx', -self.pointing_angle)
    pqr = r.as_matrix() @ xyz
    if pqr.ndim==1: pqr = np.expand_dims(pqr,axis=1)
    obj = SkyCoord(pqr.T, obstime=epoch,
            representation_type='cartesian').transform_to('icrs')
    obj.representation_type = 'spherical'
    proj = get_projection(self.center,self.scale.to_value())
    pos = np.array(obj.to_pixel(proj, origin=0))
    blocked = self.block(pos)
    pos = self.distortion(pos)

    return pd.DataFrame({
      'x': pos[0], 'y': pos[1],
      'ra': icrs.ra, 'dec': icrs.dec,
      'blocked': blocked
    })


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


  def evaluate(self, x, y):
    ''' Evaluate the source position displacement.

    Parameters:
      position (ndarray): a numpy.ndarray with the shape of (2, N(sources)).
                          the first array contains the x-coordinates, while
                          the second does the y-coordinates.
    Note:
      Not implemented yet.
    '''
    return (x,y)


@dataclass
class Detector(object):
  ''' Definition of a detector.

  Attributes:
    naxis1 (int)          : detector pixels along with NAXIS1.
    naxis2 (int)          : detector pixels along with NAXIS2.
    pixel_scale (Quantity): nominal detector pixel scale.
    offset_dx (Quantity)  : the offset along with the x-axis.
    offset_dy (Quantity)  : the offste along with the y-axis.
    position_angle (Angle): the position angle of the detector.
    displacement (PixelDisplacement):
      an instance to define the displacements of the sources due to
      the pixel non-uniformity.
  '''
  naxis1: int = 4096
  naxis2: int = 4096
  pixel_scale: Quantity = 10*u.um
  offset_dx: Quantity = 0*u.um
  offset_dy: Quantity = 0*u.um
  position_angle: Angle = Angle(0.0, unit='degree')
  displacement: PixelDisplacement = None

  def __post_init__(self):
    if self.displacement is None:
      self.displacement = PixelDisplacement()
      self.displacement.initialize(self.naxis1,self.naxis2)

  @property
  def width(self):
    ''' The physical width of the detector. '''
    return self.naxis1*self.pixel_scale.to_value(u.um)
  @property
  def height(self):
    ''' The physical height of the detector. '''
    return self.naxis2*self.pixel_scale.to_value(u.um)
  @property
  def xrange(self):
    ''' The x-axis range of the detector. '''
    return np.array((-self.width/2,self.width/2))
  @property
  def yrange(self):
    ''' The y-axis range of the detector. '''
    return np.array((-self.height/2,self.height/2))
  @property
  def footprint(self):
    ''' The footprint of the detector on the focal plane. '''
    c,s = np.cos(self.position_angle.rad),np.sin(self.position_angle.rad)
    x0,y0 = self.offset_dx.to_value(u.um),self.offset_dy.to_value(u.um)
    x1 = x0 - (self.width*c - self.height*s)/2
    y1 = y0 - (self.width*s + self.height*c)/2
    return Rectangle((x1,y1), width=self.width, height=self.height,
        angle=self.position_angle.deg, ec='r', linewidth=2, fill=False)

  def align(self, x, y):
    ''' Align the source position to the detector.

    Parameters:
      x (Series): the x-coordinates on the focal plane.
      y (Series): the y-coordinates on the focal plane.

    Return:
      The tuple of the x- and y-positions of the sources, which are remapped
      onto the detector coordinates.
    '''
    c,s = np.cos(-self.position_angle.rad),np.sin(-self.position_angle.rad)
    dx,dy = x-self.offset_dx.to_value(u.um), y-self.offset_dy.to_value(u.um)
    return c*dx-s*dy, s*dx+c*dy

  def capture(self, position):
    ''' Calculate the positions of the sources on the detector.

    Parameters:
      position (DataFrame): the positions of the sources on the focal plane.
                            the "x" and "y" columns are respectively the x-
                            and y-positions of the sources in units of micron.

    Return:
      A list of `DataFrame`s which contains the positions on the detectors.
      The number of the `DataFrame`s are the same as the detectors.
      The "x" and "y" columns are the positions on each detector. The "ra"
      and "dec" columns are the original positions in the ICRS frame.
    '''
    x,y = self.align(position.x, position.y)
    x,y = self.displacement.evaluate(x,y)
    position.x = x
    position.y = y
    bf = ~position.blocked
    xf = ((self.xrange[0] < x) & (x < self.xrange[1]))
    yf = ((self.yrange[0] < y) & (y < self.yrange[1]))
    return position.loc[xf&yf&bf,:]


@dataclass
class Telescope(object):
  ''' An imaginary telescope instance.

  The `Telescope` class is composed of an `Optics` instance and a list of
  `Detector` instances. This instance organizes the alignment of the detectors
  and converts the coordinates of the astronomical sources into the positions
  on the detectors.

  Attributes:
    pointing (SkyCoord)
    position_angle (Angle):
  '''
  pointing: SkyCoord        = None
  position_angle: Angle     = None
  optics: Optics            = None
  detectors: List[Detector] = None

  def __post_init__(self):
    if self.optics is None:
      self.optics = Optics(self.pointing, self.position_angle)
    else:
      self.pointing = self.optics.pointing
      self.position_angle = self.optics.position_angle
    if self.detectors is None:
      self.detectors = [Detector(),]
    assert self.optics is not None
    assert self.detectors is not None

  def set_distortion(self, distortion):
    ''' Set a distortion function to the optics.

    Parameters:
      distortion (function): a function to distort focal plane image.
    '''
    self.optics.set_distortion(distortion)

  def display_focal_plane(self, sources=None, epoch=None,
                          markersize=1, marker='x'):
    ''' Display the layout of the detectors.

    Show the layout of the detectors on the focal plane. The detectors are
    illustrated by the red rectangles. If the `sources` are provided, the
    detectors are overlaid on the sources on the focal plane.

    Parameters:
      sources (SkyCoord): the coordinates of astronomical sources.
      epoch (Time)      : the observation epoch.
    '''
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0)
    ax.add_patch(PolygonPatch(
      self.optics.valid_region, color=(0.8,0.8,0.8), alpha=0.2))
    if sources is not None:
      position = self.optics.imaging(sources, epoch)
      ax.scatter(position.x,position.y,markersize,marker=marker)
    for d in self.detectors:
      ax.add_patch(d.footprint)
    ax.autoscale_view()
    ax.grid()
    ax.set_xlabel('Displacement on the focal plane ($\mu$m)', fontsize=14)
    ax.set_ylabel('Displacement on the focal plane ($\mu$m)', fontsize=14)
    fig.tight_layout()
    plt.show()


  def observe(self, sources, epoch=None):
    ''' Observe astronomical sources.

    Map the sky coordinates of astronomical sources into the physical
    positions on the detectors of the telescope.

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
    fov = []
    for det in self.detectors:
      fov.append(det.capture(position))

    return fov
