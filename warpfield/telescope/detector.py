#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' Definition of Detector class '''

from dataclasses import dataclass
from typing import Callable
from astropy.coordinates import Angle
from astropy.units.quantity import Quantity
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
import astropy.units as u
import numpy as np

from .distortion import identity_transformation


@dataclass
class Detector:
    ''' Definition of a detector

    Attributes:
      naxis1 (int)           : The number of pixels along with NAXIS1.
      naxis2 (int)           : The number of pixels along with NAXIS2.
      pixel_scale (Quantity) : A nominal detector pixel scale.
      offset_dx (Quantity)   : An offset along with the x-axis.
      offset_dy (Quantity)   : An offste along with the y-axis.
      position_angle (Angle) : A position angle of the detector.
      displacement (function): A function to distort image.
    '''
    naxis1: int = 4096
    naxis2: int = 4096
    pixel_scale: Quantity = 10 * u.um
    offset_dx: Quantity = 0 * u.um
    offset_dy: Quantity = 0 * u.um
    position_angle: Angle = Angle(0.0, unit='degree')
    displacement: Callable = identity_transformation

    @property
    def width(self):
        ''' The physical width of the detector '''
        return self.naxis1 * self.pixel_scale.to_value(u.um)

    @property
    def height(self):
        ''' The physical height of the detector '''
        return self.naxis2 * self.pixel_scale.to_value(u.um)

    @property
    def xrange(self):
        ''' The x-axis range (limits) of the detector '''
        return np.array((-self.naxis1 / 2, self.naxis1 / 2))

    @property
    def yrange(self):
        ''' The y-axis range (limits) of the detector '''
        return np.array((-self.naxis2 / 2, self.naxis2 / 2))

    @property
    def detector_origin(self):
        ''' Returns the location of the lower left corner '''
        c = np.cos(self.position_angle.rad)
        s = np.sin(self.position_angle.rad)
        x0 = self.offset_dx.to_value(u.um)
        y0 = self.offset_dy.to_value(u.um)
        return [
            x0 - (self.width * c - self.height * s) / 2,
            y0 - (self.width * s + self.height * c) / 2,
        ]

    @property
    def footprint_as_patch(self):
        ''' The focal-plane footprint as a patch '''
        return Rectangle(self.detector_origin,
                         width=self.width,
                         height=self.height,
                         angle=self.position_angle.deg,
                         ec='r',
                         linewidth=2,
                         fill=False)

    @property
    def footprint_as_polygon(self):
        ''' The focal-plane footprint as a polygon '''
        c, s = np.cos(self.position_angle.rad), np.sin(self.position_angle.rad)
        x0, y0 = self.offset_dx.to_value(u.um), self.offset_dy.to_value(u.um)
        x1 = x0 - (+self.width * c - self.height * s) / 2
        y1 = y0 - (+self.width * s + self.height * c) / 2
        x2 = x0 - (-self.width * c - self.height * s) / 2
        y2 = y0 - (-self.width * s + self.height * c) / 2
        x3 = x0 - (-self.width * c + self.height * s) / 2
        y3 = y0 - (-self.width * s - self.height * c) / 2
        x4 = x0 - (+self.width * c + self.height * s) / 2
        y4 = y0 - (+self.width * s - self.height * c) / 2
        return Polygon(([x1, y1], [x2, y2], [x3, y3], [x4, y4]))

    def align(self, position):
        ''' Align the source position to the detector

        Arguments:
          position (DataFrame): The xy-coordinates on the focal plane.

        Returns:
          A numpy array of the xy-positions of the sources,
          which are remapped onto the detector coordinates.
        '''
        x, y = position.x, position.y
        c = np.cos(-self.position_angle.rad)
        s = np.sin(-self.position_angle.rad)
        dx = x - self.offset_dx.to_value(u.um)
        dy = y - self.offset_dy.to_value(u.um)
        return np.stack([
            (c * dx - s * dy) / self.pixel_scale,
            (s * dx + c * dy) / self.pixel_scale,
        ]).T

    def capture(self, position):
        ''' Calculate the positions of the sources on the detector

        Arguments:
          position (DataFrame):
              The positions of the sources on the focal plane. the "x" and "y"
              columns are respectively the x- and y-positions of the sources
              in units of micron.

        Returns:
          A list of `DataFrame`s which contains the positions on the detectors.
          The number of the `DataFrame`s are the same as the detectors.
          The "x" and "y" columns are the positions on each detector. The "ra"
          and "dec" columns are the original positions in the ICRS frame.
        '''
        xy = self.align(position)
        xy = self.displacement(xy)
        position.x = xy[:, 0]
        position.y = xy[:, 1]
        xf = ((self.xrange[0] < position.x) & (position.x < self.xrange[1]))
        yf = ((self.yrange[0] < position.y) & (position.y < self.yrange[1]))
        return position.loc[xf & yf, :]
