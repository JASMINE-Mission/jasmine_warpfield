#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' Definition of Telescope class '''

from dataclasses import dataclass
from typing import List
from astropy.coordinates import SkyCoord, Angle
import numpy as np

from .optics import Optics
from .detector import Detector
from .util import estimate_frame_from_ctype


@dataclass
class Telescope:
    ''' An imaginary telescope instance

    The `Telescope` class is composed of an `Optics` instance and a list of
    `Detector` instances. The instance organizes the alignment of the detectors
    and converts the coordinates of the astronomical sources into the positions
    on the detectors.

    Attributes:
      pointing (SkyCoord)         : A pointing direction of the telescope.
      position_angle (Angle)      : A position angle of the telescope.
      optics (Optics)             : An optics.
      detectors (List of Detector): A list of detectors.
    '''
    pointing: SkyCoord = None
    position_angle: Angle = None
    optics: Optics = None
    detectors: List[Detector] = None

    def __post_init__(self):
        if self.optics is None:
            self.optics = Optics(self.pointing, self.position_angle)
        else:
            assert self.pointing is None, \
                'pointing and optics are given at the same time.'
            self.pointing = self.optics.pointing
            self.position_angle = self.optics.position_angle
        if self.detectors is None:
            self.detectors = [Detector()]
        assert self.optics is not None, 'no optics found'
        assert self.detectors is not None, 'no detector found'

    def set_distortion(self, distortion):
        ''' Set a distortion function to the optics

        See `Optics.set_distortion` for details.

        Arguments:
          distortion (function): A function to distort focal plane image.
        '''
        self.optics.set_distortion(distortion)

    def get_footprints(self, frame, **options):
        ''' Obtain detector footprints on the sky

        Options:
          limit (bool):
              Limit the footprints within the valid region if True.
        '''
        limit = options.pop('limit', True)

        footprints = []
        field_of_view = self.optics.field_of_view
        proj = self.optics.get_projection(frame)

        for d in self.detectors:
            fp = field_of_view.intersection(
                d.footprint_as_polygon) if limit else d.footprint_as_polygon
            edge = np.array(fp.boundary.coords)
            sky = np.array(proj.all_pix2world(edge, 0))
            # sky = SkyCoord(pos[:, 0] * u.deg, pos[:, 1] * u.deg, frame=frame)
            # if frame == 'galactic':
            #     sky = sky.galactic
            #     pos = Polygon(np.stack([sky.l.deg, sky.b.deg]).T)
            # else:
            #     sky = sky.icrs
            #     pos = Polygon(np.stack([sky.ra.deg, sky.dec.deg]).T)
            footprints.append(sky)
        return footprints

    def overlay_footprints(self, axis, **options):
        ''' Display the footprints on the given axis

        Arguments:
          axis (WCSAxesSubplot):
            An axis instance with a WCS projection.

        Options:
          label (string): The label of the footprints.
          color (Color): Color of the footprint edges.
        '''
        assert hasattr(axis, 'wcs'), \
            'axis should be an instance of WCSAxesSubplot'

        label = options.pop('label', None)
        color = options.pop('color', 'C2')
        frame = estimate_frame_from_ctype(axis.wcs.wcs.ctype)
        proj = axis.wcs

        for footprint in self.get_footprints(frame, **options):
            v = np.array(proj.all_world2pix(footprint, 0))
            # v = np.array(footprint.boundary.coords)
            axis.plot(v[:, 0], v[:, 1], c=color, label=label, **options)
        return axis

    def display_focal_plane(
            self, axis, sources=None, epoch=None, **options):
        ''' Display the layout of the detectors

        Show the layout of the detectors on the focal plane. The detectors are
        illustrated by the red rectangles. If the `sources` are provided, the
        detectors are overlaid on the sources on the focal plane.

        Arguments:
          axis (Axes)       : a Matplotlib Axes instance.
          sources (SkyCoord): A list of astronomical sources.
          epoch (Time)      : The observation epoch.

        Options:
          figsize (tuple(int,int)): The figure size.
          marker (string): A marker style to show sources.
          markersize (float): The size of markers.
        '''
        markersize = options.pop('markersize', 1)
        marker = options.pop('marker', 'x')
        color = options.pop('color', (0.8, 0.8, 0.8))
        alpha = options.pop('alpha', 0.2)
        axis.set_aspect(1.0)
        axis.add_patch(self.optics.get_fov_patch(color=color, alpha=alpha))
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

    def observe(self, sources, epoch=None):
        ''' Observe astronomical sources

        Map the sky coordinates of astronomical sources into the physical
        positions on the detectors of the telescope.

        Arguments:
          sources (SkyCoord): A list of astronomical sources.
          epoch (Time): The datetime of the observation.

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
