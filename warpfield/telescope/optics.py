#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' Definition of Optics class '''

import sys

from dataclasses import dataclass
from typing import Callable
from astropy.coordinates import SkyCoord, Angle
from astropy.units.quantity import Quantity
from shapely.geometry import Polygon, Point
from shapely.geometry import MultiPoint
from shapely.prepared import prep
from matplotlib.patches import Polygon as PolygonPatch
import astropy.units as u
import numpy as np
import pandas as pd

from .util import get_projection
from .distortion import identity_transformation


@dataclass
class Optics:
    ''' Definition of optical components

    Attributes:
      pointing (SkyCoord)    : The latitude of the telescope pointing.
      position_angle (Angle) : The position angle of the telescope.
      focal_length (Quantity): The focal length of the telescope in meter.
      diameter (Quantity)    : The diameter of the telescope in meter.
      field_of_view (Polygon) : The valid region of the focal plane.
      margin (Quantity)      : The margin of the valid region (buffle).
      distortion (function)  : A function to distort the focal plane image.
    '''
    pointing: SkyCoord
    position_angle: Angle = Angle(0.0, unit='degree')
    focal_length: Quantity = 4.86 * u.m
    diameter: Quantity = 0.4 * u.m
    field_of_view: Polygon = Point(0, 0).buffer(30000)
    margin: Quantity = 5000 * u.um
    distortion: Callable = identity_transformation

    @property
    def scale(self):
        ''' A conversion factor from sky to focal plane in degree/um '''
        return (1.0 * u.rad / self.focal_length).to(u.deg / u.um)

    @property
    def center(self):
        ''' A dummy position to defiine the center of the focal plane '''
        return SkyCoord(0 * u.deg, 0 * u.deg, frame='icrs')

    @property
    def pointing_angle(self):
        ''' Angles to define the pointing position and orientation '''
        # use the ICRS frame in calculation.
        icrs = self.pointing.icrs
        # calculate position angle in the ICRS frame.
        north = self.pointing.directional_offset_by(0.0, 1 * u.arcsec)
        delta = self.pointing.icrs.position_angle(north)
        position_angle = -self.position_angle.rad - delta.rad
        return np.array((icrs.ra.rad, -icrs.dec.rad, position_angle))

    @property
    def projection(self):
        ''' Get the projection of the optics '''
        return self.get_projection(self.pointing.frame)

    def get_projection(self, frame):
        ''' Obtain the projection with a specific frame'''
        reference = self.pointing.transform_to(frame)
        position_angle = self.get_position_angle(frame)
        return get_projection(
            reference, self.scale.to_value(), position_angle)

    def get_position_angle(self, frame):
        ''' Obtain the position angle for a specific frame'''
        origin = self.pointing.transform_to(frame)
        original = self.pointing.directional_offset_by(0.0, 1 * u.arcsec)
        delta = origin.position_angle(original)
        return self.position_angle + delta

    def get_polygon(self, **options):
        ''' Get a patch of the field of view for Matplotlib '''
        xy = self.field_of_view.exterior.xy
        return PolygonPatch(np.array(xy).T, **options)

    def set_distortion(self, distortion):
        ''' Assign a distortion function

        The argument of the distortion function should be a numpy.array with
        the shape of (2, Nsrc). The first element contains the x-positions,
        while the second element contains the y-positions.

        Arguments:
          distortion (function): A function to distort focal plane image.
        '''
        self.distortion = distortion

    def block(self, position):
        ''' Block sources outside the field of view

        Arguments:
          position (ndarray):
              Source positions on the focal plane w/o distortion.

        Returns:
          A boolean array, where True if a source is located inside
          the field-of-view.
        '''
        mp = MultiPoint(position.reshape((2, -1)).T)
        polygon = prep(self.field_of_view.buffer(self.margin.to_value(u.um)))
        return np.array([not polygon.contains(p) for p in mp.geoms])

    def imaging(self, sources, epoch=None):
        ''' Map celestial positions onto the focal plane

        Arguments:
          sources (SkyCoord): The coordinates of sources.
          epoch (Time): The epoch of the observation.

        Returns:
          A `DataFrame` instance.
          The DataFrame contains four columns: the "x" and "y" columns are
          the positions on the focal plane in micron, and the "ra" and "dec"
          columns are the original celestial positions in the ICRS frame.
        '''
        try:
            if epoch is not None:
                sources = sources.apply_space_motion(epoch)
        except Exception as e:
            print(str(e), file=sys.stderr)
            print('No proper motion information is available.',
                  file=sys.stderr)
            print('The positions are not updated to new epoch.',
                  file=sys.stderr)
        # icrs = sources.transform_to('icrs')
        # xyz = icrs.cartesian.xyz
        # r = Rotation.from_euler('zyx', -self.pointing_angle)
        # pqr = np.atleast_2d(r.as_matrix() @ xyz)
        # obj = SkyCoord(pqr.T, obstime=epoch,
        #                representation_type='cartesian').transform_to('icrs')
        # obj.representation_type = 'spherical'
        # pos = np.array(obj.to_pixel(self.projection, origin=0))

        pos = sources.to_pixel(self.projection, 0)
        pos = np.array(pos).reshape((2, -1))
        blocked = self.block(pos)
        pos = self.distortion(pos.copy())

        return pd.DataFrame({
            'x': pos[0],
            'y': pos[1],
            'ra': sources.icrs.ra,
            'dec': sources.icrs.dec,
        })[~blocked]
