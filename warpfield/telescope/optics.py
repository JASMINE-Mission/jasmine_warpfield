#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' Definition of Optics class '''

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

from .util import get_projection
from .source import SourceTable
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

    def contains(self, position):
        ''' Check if sources are inside the field of view

        Arguments:
          position (ndarray):
              Source positions on the focal plane w/o distortion.

        Returns:
          A boolean array, where True if a source is located inside
          the field-of-view.
        '''
        mp = MultiPoint(position.reshape((2, -1)).T)
        polygon = prep(self.field_of_view.buffer(self.margin.to_value(u.um)))
        return np.array([polygon.contains(p) for p in mp.geoms])

    def imaging(self, sources, epoch=None):
        ''' Map celestial positions onto the focal plane

        Arguments:
          sources (SourceTable): A `SourceTable` instance
          epoch (Time): The epoch of the observation.

        Returns:
          A `SourceTable` instance with positions on the focal plane.
        '''
        temp = SourceTable(sources.table)
        if epoch is not None:
            temp.apply_space_motion(epoch)
        skycoord = temp.skycoord

        pos = skycoord.to_pixel(self.projection, 0)
        pos = np.array(pos).reshape((2, -1))
        within_fov = self.contains(pos)
        pos = self.distortion(pos.copy())

        table = temp.table
        table['x'] = pos[0] * u.um
        table['y'] = pos[1] * u.um
        table = table[within_fov]

        return SourceTable(table=table)
