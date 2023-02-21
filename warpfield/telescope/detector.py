#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' Definition of Detector class '''

from dataclasses import dataclass
from typing import Callable
from astropy.table import QTable
from astropy.coordinates import Angle
from astropy.units.quantity import Quantity
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from shapely.geometry import Polygon
import astropy.units as u
import numpy as np

from .source import DetectorPositionTable
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
        return self.naxis1 * self.pixel_scale.to(u.um)

    @property
    def height(self):
        ''' The physical height of the detector '''
        return self.naxis2 * self.pixel_scale.to(u.um)

    @property
    def xrange(self):
        ''' The x-axis range (limits) of the detector '''
        return np.array((-self.naxis1 / 2, self.naxis1 / 2))

    @property
    def yrange(self):
        ''' The y-axis range (limits) of the detector '''
        return np.array((-self.naxis2 / 2, self.naxis2 / 2))

    @property
    def corners(self):
        ''' The detector corner positions in units of micron '''
        cos = np.cos(self.position_angle.rad)
        sin = np.sin(self.position_angle.rad)
        width = self.width.to_value(u.um)
        height = self.height.to_value(u.um)
        x0 = self.offset_dx.to_value(u.um)
        y0 = self.offset_dy.to_value(u.um)
        x1 = x0 - (+width * cos - height * sin) / 2
        y1 = y0 - (+width * sin + height * cos) / 2
        x2 = x0 - (-width * cos - height * sin) / 2
        y2 = y0 - (-width * sin + height * cos) / 2
        x3 = x0 - (-width * cos + height * sin) / 2
        y3 = y0 - (-width * sin - height * cos) / 2
        x4 = x0 - (+width * cos + height * sin) / 2
        y4 = y0 - (+width * sin - height * cos) / 2
        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] * u.um

    @property
    def detector_origin(self):
        ''' Returns the location of the lower left corner (origin) '''
        return self.corners[0]

    def within(self, axis, position):
        ''' Check if positions are within the detector range

        Arguments:
          axis (str):
              The name of the axis ('x' or 'y').
          position (QTable):
              Table with 'nx' and 'ny' columns.

        Returns:
          True if positions are within `self.xrange` or `self.yrange`.
        '''
        assert axis in ('x', 'y'), '`axis` should be "x" or "y".'
        range = self.xrange if axis == 'x' else self.yrange
        return (range[0] <= position) & (position <= range[1])

    def get_footprint_as_patch(self, **options):
        ''' Returns a focal-plane footprint as a patch

        Options:
          edgecolor: default='r'
          linewidth: default=2
          fill: default=False

        Returns:
          A `Rectangle` instance for Matplotlib.
          The origin of the axes should be the telescope's optical center.
          The units of the axes should be micron.
        '''
        options['edgecolor'] = options.get('edgecolor', 'r')
        options['linewidth'] = options.get('linewidth', 2)
        options['fill'] = options.get('fill', False)
        return Rectangle(
            self.detector_origin.to_value(u.um),
            width=self.width.to_value(u.um),
            height=self.height.to_value(u.um),
            angle=self.position_angle.deg,
            **options)

    def get_first_line_as_patch(self, **options):
        ''' Returns the detector's first line as a patch

        Options:
          linewidth: default=4.0
          color: default='b'
          alpha: default=0.5

        Returns:
          A `Line2D` instance for MatplotLib.
          The origin of the axes should be the telescope's optical center.
          The units of the axes should be micron.
        '''
        options['linewidth'] = options.get('linewidth', 4.0)
        options['color'] = options.get('color', 'b')
        options['alpha'] = options.get('alpha', 0.5)
        xdata = self.corners[0:2, 0]
        ydata = self.corners[0:2, 1]
        return Line2D(xdata, ydata, **options)

    def get_footprint_as_polygon(self, **options):
        ''' The focal-plane footprint as a polygon

        Returns:
          A `Polygon` object for Shapely.
          The origin of the canvas should be the tehescope's optical center.
          The units of the canvas should be micron.
        '''
        return Polygon(self.corners.to_value(u.um), **options)

    def align(self, position):
        ''' Align the source position to the detector

        Arguments:
          position (QTable):
              The (x,y)-coordinates on the focal plane.

        Returns:
          A QTable instance of the positions of the sources,
          which are remapped onto the detector coordinates.
        '''
        x, y = position['x'], position['y']
        c = np.cos(-self.position_angle.rad)
        s = np.sin(-self.position_angle.rad)
        dx = x - self.offset_dx
        dy = y - self.offset_dy
        return QTable([
            (c * dx - s * dy) / self.pixel_scale,
            (s * dx + c * dy) / self.pixel_scale,
        ], names=['nx', 'ny'])

    def contains(self, position):
        ''' Return True if objects are on the detector

        Argument:
          position (QTable):
              The (nx,ny)-coordinates on the detector.

        Returns:
          A boolean array.
          True if positions are on the detector.
        '''
        xf = self.within('x', position['nx'])
        yf = self.within('y', position['ny'])
        return xf & yf

    def capture(self, position):
        ''' Calculate the positions of the sources on the detector

        Arguments:
          position (FocalPlaneTable):
              The positions of the sources on the focal plane. the "x" and "y"
              columns are respectively the x- and y-positions of the sources
              in units of micron.

        Returns:
          A `DetectorPositionTable`.
          The "nx" and "ny" columns are the positions on each detector.
        '''
        table = position.table.copy()
        xy = self.displacement(self.align(table))
        table['nx'] = xy['nx']
        table['ny'] = xy['ny']
        within = self.contains(table)
        return DetectorPositionTable(table[within])
