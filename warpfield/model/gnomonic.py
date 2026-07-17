#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Simple distortion-free gnomonic system model '''

import astropy.units as u
import numpy as np

from ..detector import Detector
from ..distortion import Distortion, IdentityDistortion
from ..projection import GnomonicProjection
from ..simulator import Simulator
from ..telescope import Telescope


__all__ = ['get_gnomonic']


def _get_gnomonic(distortion, simulator):
    ''' Construct the common simple gnomonic system '''
    if not isinstance(distortion, Distortion):
        raise TypeError('`distortion` should be a Distortion instance.')
    if not isinstance(simulator, bool):
        raise TypeError('`simulator` should be a boolean.')

    detector_shape = (4096, 4096)
    pixel_size = 10 * u.um
    angular_pixel_scale = 1 * u.arcsec
    focal_plane_scale = (
        pixel_size / angular_pixel_scale
    ).to_value(u.mm / u.deg)

    detector = Detector(
        rotation=0.0,
        offset=[0.0, 0.0],
        pixel_scale=np.full(2, pixel_size.to_value(u.mm)),
        shape=detector_shape,
    )
    telescope_type = Simulator if simulator else Telescope
    return telescope_type(
        projection=GnomonicProjection(),
        distortion=distortion,
        plate_scale=np.full(2, focal_plane_scale),
        detectors=(detector,),
    )


def get_gnomonic(simulator=False):
    ''' Return a simple distortion-free gnomonic system

    The centered 4096 x 4096 detector has 10 um pixels. The angular scale is
    1 arcsec/pixel, represented internally as 36 mm/degree.
    '''
    return _get_gnomonic(IdentityDistortion(), simulator)
