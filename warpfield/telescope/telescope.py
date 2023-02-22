#!/usri/bin/env python
# -*- coding: utf-8 -*-
''' Definition of Telescope class '''

from dataclasses import dataclass
from typing import List
from astropy.coordinates import SkyCoord, Angle
from astropy.table import vstack
import numpy as np

from .source import DetectorPositionTable
from .optics import Optics
from .detector import Detector
from .source import convert_skycoord_to_sourcetable
from .util import estimate_frame_from_ctype


@dataclass
class Telescope:
    ''' An imaginary telescope instance

    The `Telescope` class is composed of an `Optics` instance and a list of
    `Detector` instances. The instance organizes the alignment of the detectors
    and converts the coordinates of the astronomical sources into the positions
    on the detectors.

    Attributes:
      pointing (SkyCoord):
          A pointing direction of the telescope.
      position_angle (Angle):
          A position angle of the telescope.
      optics (Optics):
          An `Optics` instance.
      detectors (List of Detector):
          A list of `Detector` instances.
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

    def get_footprints(self, frame, **options):
        ''' Obtain detector footprints on the sky

        Options:
          limit (bool):
              Limit the footprints within the valid region if True.

        Returns:
          A list of detector footprints on the sky.
          Each footprint is given as a 2-dimensional numpy array [[x,y], ...].
        '''
        limit = options.pop('limit', True)

        footprints = []
        fov = self.optics.field_of_view
        proj = self.optics.get_projection(frame)

        for d in self.detectors:
            if limit:
                fp = fov.intersection(d.get_footprint_as_polygon())
            else:
                fp = d.get_footprint_as_polygon()
            edge = np.array(fp.boundary.coords)
            sky = np.array(proj.all_pix2world(edge, 0))
            footprints.append(sky)
        return footprints

    def overlay_footprints(self, axis, **options):
        ''' Overlay the detector footprints on the sky

        Display the footprints of the detectors on the given canvas.
        Note that this function does not take into account the distortion.

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
            axis.plot(v[:, 0], v[:, 1], c=color, label=label, **options)
        return axis

    def display_focal_plane(
            self, axis, source=None, epoch=None, **options):
        ''' Display the layout of the detectors

        Show the layout of the detectors on the focal plane. The detectors are
        illustrated by the red rectangles. If the `sources` are provided, the
        detectors are overlaid on the sources on the focal plane.

        Arguments:
          axis (Axes):
              a Matplotlib Axes instance.
          source (SkyCoord or SourceTable):
              A list of astronomical sources.
          epoch (Time):
              The observation epoch.

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
        if source is not None:
            if isinstance(source, SkyCoord):
                source = convert_skycoord_to_sourcetable(source)
            fp = self.optics.imaging(source, epoch).table
            axis.scatter(fp['x'], fp['y'], markersize, marker=marker)
        for d in self.detectors:
            axis.add_patch(d.get_footprint_as_patch())
            axis.add_patch(d.get_first_line_as_patch())
        axis.autoscale_view()
        axis.grid()
        axis.set_xlabel(r'Displacement on the focal plane ($\mu$m)',
                        fontsize=14)
        axis.set_ylabel(r'Displacement on the focal plane ($\mu$m)',
                        fontsize=14)

    def observe(self, source, epoch=None, stack=False):
        ''' Observe astronomical sources

        Map the sky coordinates of astronomical sources into the physical
        positions on the detectors of the telescope.

        Arguments:
          sources (SourceTable):
              A list of astronomical sources.

        Options:
          epoch (Time):
              The datetime of the observation.
          stack (bool):
              A stacked table is returned if true.

        Returns:
          A list of DetectorPlaneTable, with the shape of N(detector).
          The first index specifies the detector of the telescope.
          All tables are stacked into a single table if `stack` is True.
        '''
        fp_position = self.optics.imaging(source, epoch)
        dets = []
        for n, det in enumerate(self.detectors):
            det_position = det.capture(fp_position)
            det_position.table['detector'] = n
            dets.append(det_position)

        if stack is False:
            return dets
        else:
            stacked = vstack([d.table for d in dets])
            return DetectorPositionTable(stacked)
