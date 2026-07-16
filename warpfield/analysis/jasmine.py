#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Nominal JASMINE telescope preset '''

import astropy.units as u
import numpy as np

from .detector import Detector
from .distortion import Distortion, IdentityDistortion
from .optics import Optics
from .projection import GnomonicProjection
from .simulator import Simulator
from .telescope import Telescope
from .utils import plate_scale


__all__ = ['get_jasmine']


def get_jasmine(distortion=None, simulator=True):
    ''' Return a Telescope configured for the nominal JASMINE focal plane

    A masked ``Simulator`` is returned by default. Set ``simulator=False`` to
    return the corresponding unmasked ``Telescope``.
    '''
    if distortion is None:
        distortion = IdentityDistortion()
    if not isinstance(distortion, Distortion):
        raise TypeError('`distortion` should be a Distortion instance.')
    if not isinstance(simulator, bool):
        raise TypeError('`simulator` should be a boolean.')

    detector_shape = (1920, 1920)
    pixel_size = 10 * u.um
    sensor_gap = 3.0 * u.mm
    focal_length = 4.86 * u.m

    half_step = (
        detector_shape[0] * pixel_size / 2
        + sensor_gap / 2
    ).to_value(u.mm)
    imaging_half_width = (
        detector_shape[0] * pixel_size
        + sensor_gap / 2
    ).to_value(u.mm)
    imaging_radius = float(np.sqrt(2) * imaging_half_width)

    optics = Optics(
        projection=GnomonicProjection(),
        distortion=distortion,
        plate_scale=plate_scale(focal_length),
        imaging_radius=imaging_radius,
    )
    offsets = (
        (-half_step, -half_step),
        (+half_step, -half_step),
        (+half_step, +half_step),
        (-half_step, +half_step),
    )
    rotations = (0.0, 90.0, 180.0, 270.0)
    detectors = tuple(
        Detector(
            rotation=rotation,
            offset=offset,
            pixel_scale=np.full(
                2,
                pixel_size.to_value(u.mm),
            ),
            shape=detector_shape,
        )
        for rotation, offset in zip(rotations, offsets)
    )
    telescope_type = Simulator if simulator else Telescope
    return telescope_type(optics, detectors)
