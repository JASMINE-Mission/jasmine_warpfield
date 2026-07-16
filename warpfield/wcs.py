#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Astropy WCS interfaces for astrometric analysis '''

from astropy.wcs import WCS
import jax.numpy as jnp
import numpy as np

from .exposure import Exposure
from .projection import EquidistantProjection, GnomonicProjection
from .telescope import Telescope


__all__ = ['generate_wcs']


def _rotation_matrix(angle):
    ''' Return a two-dimensional rotation matrix for an angle in degrees '''
    angle = np.deg2rad(angle)
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), +np.cos(angle)],
    ])


def _projection_code(projection):
    if isinstance(projection, GnomonicProjection):
        return 'TAN'
    if isinstance(projection, EquidistantProjection):
        return 'ARC'
    raise TypeError('Unsupported Projection type for WCS conversion.')


def _generate_one_wcs(pointing, scale, detector, projection_code):
    ''' Generate one nominal detector WCS '''
    rotation = _rotation_matrix(float(detector.rotation))
    pixel_scale = np.asarray(detector.pixel_scale)
    offset = np.asarray(detector.offset)

    optical_center = (
        rotation @ (-offset)
    ) / pixel_scale
    focal_to_plane = (
        _rotation_matrix(float(pointing.position_angle[0]))
        @ np.diag(1 / scale)
    )
    detector_to_focal = (
        _rotation_matrix(-float(detector.rotation))
        @ np.diag(pixel_scale)
    )

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = optical_center + 1
    wcs.wcs.crval = [
        float(pointing.ra[0]),
        float(pointing.dec[0]),
    ]
    wcs.wcs.ctype = [
        f'RA---{projection_code}',
        f'DEC--{projection_code}',
    ]
    wcs.wcs.cunit = ['deg', 'deg']
    wcs.wcs.cd = (
        np.diag([-1.0, 1.0])
        @ focal_to_plane
        @ detector_to_focal
    )
    return wcs


def generate_wcs(exposure, telescope):
    ''' Generate nominal WCS objects for all exposures and detectors

    The returned nested tuple has shape ``(N_exposure, N_detector)``.
    Optical and detector distortions are intentionally excluded.
    '''
    if not isinstance(exposure, Exposure):
        raise TypeError('`exposure` should be an Exposure instance.')
    if not isinstance(telescope, Telescope):
        raise TypeError('`telescope` should be a Telescope instance.')

    projection_code = _projection_code(telescope.optics.projection)
    indices = jnp.arange(len(exposure))
    scale_factors = np.asarray(
        exposure.calibration.scale_factor(indices)
    )
    scales = (
        np.asarray(telescope.optics.plate_scale)[None, :]
        * scale_factors
    )
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError(
            'The effective plate scales should be finite and nonzero.')

    rows = []
    for index, scale in enumerate(scales):
        pointing = exposure.pointing[index]
        rows.append(tuple(
            _generate_one_wcs(
                pointing,
                scale,
                detector,
                projection_code,
            )
            for detector in telescope.detectors
        ))
    return tuple(rows)
