#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' definition of Telescope class '''

import sys

from dataclasses import dataclass
from typing import Callable, List
from astropy.coordinates import SkyCoord, Angle
from astropy.units.quantity import Quantity
from astropy.visualization.wcsaxes import WCSAxesSubplot
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, Point
from shapely.geometry import MultiPoint
from shapely.prepared import prep
from descartes.patch import PolygonPatch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd

from .util import get_projection
from .distortion import identity_transformation


@dataclass
class Optics:
    ''' definition of optical components

    Attributes:
      pointing (SkyCoord)    : the latitude of the telescope pointing.
      position_angle (Angle) : the position angle of the telescope.
      focal_length (Quantity): the focal length of the telescope in meter.
      diameter (Quantity)    : the diameter of the telescope in meter.
      field_of_view (Polygon) : the valid region of the focal plane.
      margin (Quantity)      : the margin of the valid region (buffle).
      distortion (function)  : a function to distort the focal plane image.
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
        ''' a conversion factor from sky to focal plane in degree/um '''
        return (1.0 * u.rad / self.focal_length).to(u.deg / u.um)

    @property
    def center(self):
        ''' a dummy position to defiine the center of the focal plane '''
        return SkyCoord(0 * u.deg, 0 * u.deg, frame='icrs')

    @property
    def pointing_angle(self):
        ''' angles to define the pointing position and orientation '''
        ## use the ICRS frame in calculation.
        icrs = self.pointing.icrs
        ## calculate position angle in the ICRS frame.
        north = self.pointing.directional_offset_by(0.0, 1 * u.arcsec)
        delta = self.pointing.icrs.position_angle(north)
        position_angle = -self.position_angle.rad - delta.rad
        return np.array((icrs.ra.rad, -icrs.dec.rad, position_angle))

    def set_distortion(self, distortion):
        ''' assign a distortion function

        The argument of the distortion function should be a numpy.array with
        the shape of (2, Nsrc). The first element contains the x-positions,
        while the second element contains the y-positions.

        Arguments:
          distortion (function): a function to distort focal plane image.
        '''
        self.distortion = distortion

    def block(self, position):
        ''' block sources outside the field of view

        Arguments:
          position (ndarray):
              Source positions on the focal plane w/o distortion.

        Returns:
          A boolean array, where True if a source is lcated inside the field-of-view.
        '''
        mp = MultiPoint(position.reshape((-1,2)))
        polygon = prep(self.field_of_view.buffer(self.margin.to_value(u.um)))
        return np.array([not polygon.contains(p) for p in mp.geoms])

    def imaging(self, sources, epoch=None):
        ''' map celestial positions onto the focal plane

        Arguments:
          sources (SkyCoord): the coordinates of sources.
          epoch (Time): the epoch of the observation.

        Returns:
          A `DataFrame` instance.
          The DataFrame contains four columns: the "x" and "y" columns are
          the positions on the focal plane in micron, and the "ra" and "dec"
          columns are the original celestial positions in the ICRS frame.
          The "blocked" column indicates if the sources are located within
          the field of view or not.
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
        icrs = sources.transform_to('icrs')
        xyz = icrs.cartesian.xyz
        r = Rotation.from_euler('zyx', -self.pointing_angle)
        pqr = np.atleast_2d(r.as_matrix() @ xyz)
        obj = SkyCoord(pqr.T, obstime=epoch,
                       representation_type='cartesian').transform_to('icrs')
        obj.representation_type = 'spherical'
        proj = get_projection(self.center, self.scale.to_value())
        pos = np.array(obj.to_pixel(proj, origin=0))
        pos = self.distortion(pos.copy())

        return pd.DataFrame({
            'x': pos[0],
            'y': pos[1],
            'ra': icrs.ra,
            'dec': icrs.dec,
            'blocked': self.block(pos)
        })


@dataclass
class Detector:
    ''' definition of a detector

    Attributes:
      naxis1 (int)           : detector pixels along with NAXIS1.
      naxis2 (int)           : detector pixels along with NAXIS2.
      pixel_scale (Quantity) : nominal detector pixel scale.
      offset_dx (Quantity)   : the offset along with the x-axis.
      offset_dy (Quantity)   : the offste along with the y-axis.
      position_angle (Angle) : the position angle of the detector.
      displacement (function): a function to distort image.
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
        ''' the physical width of the detector '''
        return self.naxis1 * self.pixel_scale.to_value(u.um)

    @property
    def height(self):
        ''' the physical height of the detector '''
        return self.naxis2 * self.pixel_scale.to_value(u.um)

    @property
    def xrange(self):
        ''' the x-axis range of the detector '''
        return np.array((-self.naxis1 / 2, self.naxis1 / 2))

    @property
    def yrange(self):
        ''' the y-axis range of the detector '''
        return np.array((-self.naxis2 / 2, self.naxis2 / 2))

    @property
    def detector_origin(self):
        ''' returns the location of the lower left corner '''
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
        ''' the footprint of the detector on the focal plane as a patch '''
        return Rectangle(self.detector_origin,
                         width=self.width,
                         height=self.height,
                         angle=self.position_angle.deg,
                         ec='r',
                         linewidth=2,
                         fill=False)

    @property
    def footprint_as_polygon(self):
        ''' the footprint of the detector on the focal plane '''
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
        ''' align the source position to the detector

        Arguments:
          position (DataFrame): the xy-coordinates on the focal plane.

        Returns:
          A numpy array of the xy-positions of the sources,
          which are remapped onto the detector coordinates.
        '''
        x,y = position.x, position.y
        c = np.cos(-self.position_angle.rad)
        s = np.sin(-self.position_angle.rad)
        dx = x - self.offset_dx.to_value(u.um)
        dy = y - self.offset_dy.to_value(u.um)
        return np.stack([
            (c * dx - s * dy) / self.pixel_scale,
            (s * dx + c * dy) / self.pixel_scale,
        ]).T

    def capture(self, position):
        ''' calculate the positions of the sources on the detector

        Arguments:
          position (DataFrame):
              the positions of the sources on the focal plane. the "x" and "y"
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
        position.x = xy[:,0]
        position.y = xy[:,1]
        bf = ~position.blocked
        xf = ((self.xrange[0] < position.x) & (position.x < self.xrange[1]))
        yf = ((self.yrange[0] < position.y) & (position.y < self.yrange[1]))
        return position.loc[xf & yf & bf, :]


@dataclass
class Telescope:
    ''' an imaginary telescope instance

    The `Telescope` class is composed of an `Optics` instance and a list of
    `Detector` instances. This instance organizes the alignment of the detectors
    and converts the coordinates of the astronomical sources into the positions
    on the detectors.

    Attributes:
      pointing (SkyCoord)
      position_angle (Angle):
      optics (Optics)
      detectors (List of Detector)
    '''
    pointing: SkyCoord = None
    position_angle: Angle = None
    optics: Optics = None
    detectors: List[Detector] = None

    def __post_init__(self):
        if self.optics is None:
            self.optics = Optics(self.pointing, self.position_angle)
        else:
            self.pointing = self.optics.pointing
            self.position_angle = self.optics.position_angle
        if self.detectors is None:
            self.detectors = [Detector()]
        assert self.optics is not None
        assert self.detectors is not None

    def set_distortion(self, distortion):
        ''' set a distortion function to the optics

        See `Optics.set_distortion` for details.

        Arguments:
          distortion (function): a function to distort focal plane image.
        '''
        self.optics.set_distortion(distortion)

    def get_footprints(self, **options):
        ''' obtain detector footprints on the sky

        Options:
          frame (string): specify the coordinate of the footprint.
          limit (bool): limit the footprints within the valid region.
          patch (bool): obtain PolygonPatch instead of Polygon.
        '''
        frame = options.pop('frame', self.pointing.frame.name)
        limit = options.pop('limit', True)
        patch = options.pop('patch', False)
        if self.pointing.frame.name == 'galactic':
            l0 = self.pointing.galactic.l
            b0 = self.pointing.galactic.b
        else:
            l0 = self.pointing.icrs.ra
            b0 = self.pointing.icrs.dec

        def generate(e):
            frame = self.pointing.frame

            def func(x):
                pos = x.reshape((-1, 2))
                p0 = SkyCoord(pos[:, 0], pos[:, 1], frame=frame, unit=u.deg)
                res = self.optics.imaging(p0)
                return (e - res[['x', 'y']].to_numpy()).flatten()

            return func

        footprints = []
        field_of_view = self.optics.field_of_view
        for d in self.detectors:
            fp = field_of_view.intersection(
                d.footprint_as_polygon) if limit else d.footprint_as_polygon
            edge = np.array(fp.boundary.coords)[0:-1]
            p0 = np.tile([l0.deg, b0.deg], edge.shape[0])
            func = generate(edge)
            res = least_squares(func, p0)
            pos = res.x.reshape((-1, 2))
            sky = SkyCoord(pos[:, 0] * u.deg,
                           pos[:, 1] * u.deg,
                           frame=self.pointing.frame.name)
            if frame == 'galactic':
                sky = sky.galactic
                pos = Polygon(np.stack([sky.l.deg, sky.b.deg]).T)
            else:
                sky = sky.icrs
                pos = Polygon(np.stack([sky.ra.deg, sky.dec.deg]).T)
            footprints.append(PolygonPatch(pos, **options) if patch else pos)
        return footprints

    def overlay_footprints(self, axis, **options):
        ''' display the footprints on the given axis

        Arguments:
          axis (WCSAxesSubplot):
            An axis instance with a WCS projection.

        Options:
          frame (string): the coodinate frame.
          label (string): the label of the footprints.
          color (Color): color of the footprint edges.
        '''
        label = options.pop('label', None)
        color = options.pop('color', 'C2')
        frame = options.pop('frame', self.pointing.frame.name)
        if hasattr(axis, 'get_transform'):
            options['transform'] = axis.get_transform(frame)
        for footprint in self.get_footprints(frame=frame, **options):
            v = np.array(footprint.boundary.coords)
            axis.plot(v[:, 0], v[:, 1], c=color, label=label, **options)
        return axis

    def display_focal_plane(self,
                            sources=None,
                            epoch=None,
                            axis=None,
                            **options):
        ''' display the layout of the detectors

        Show the layout of the detectors on the focal plane. The detectors are
        illustrated by the red rectangles. If the `sources` are provided, the
        detectors are overlaid on the sources on the focal plane.

        Arguments:
          sources (SkyCoord): the coordinates of astronomical sources.
          epoch (Time)      : the observation epoch.

        Options:
          figsize (tuple(int,int)): the figure size.
          marker (string): marker style to show sources.
          markersize (float): size of markers.
        '''
        markersize = options.pop('markersize', 1)
        marker = options.pop('marker', 'x')
        figsize = options.pop('figsize', (8, 8))
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(111)
        axis.set_aspect(1.0)
        axis.add_patch(
            PolygonPatch(self.optics.field_of_view,
                         color=(0.8, 0.8, 0.8),
                         alpha=0.2))
        if sources is not None:
            position = self.optics.imaging(sources, epoch)
            axis.scatter(position.x, position.y, markersize, marker=marker)
        for d in self.detectors:
            axis.add_patch(d.footprint_as_patch)
        axis.autoscale_view()
        axis.grid()
        axis.set_xlabel(r'Displacement on the focal plane ($\mu$m)',
                        fontsize=14)
        axis.set_ylabel(r'Displacement on the focal plane ($\mu$m)',
                        fontsize=14)
        if axis is None: fig.tight_layout()

    def observe(self, sources, epoch=None):
        ''' observe astronomical sources

        Map the sky coordinates of astronomical sources into the physical
        positions on the detectors of the telescope.

        Arguments:
          sources (SkyCoord): a list of astronomical sources.
          epoch (Time): the datetime of the observation.

        Returns:
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
