#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Nominal detector and telescope footprints '''

from operator import index as integer_index

from astropy.coordinates import SkyCoord
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Point, Polygon

from .detector import Detector
from .exposure import Exposure
from .projection import EquidistantProjection, GnomonicProjection
from .telescope import Telescope


__all__ = ['detector_footprint', 'telescope_footprints']


def _rectangle_boundary(shape, samples_per_edge):
    ''' Generate a closed, counterclockwise detector boundary '''
    half = np.asarray(shape, dtype=float) / 2
    corners = np.array([
        [-half[0], -half[1]],
        [+half[0], -half[1]],
        [+half[0], +half[1]],
        [-half[0], +half[1]],
    ])
    fraction = np.arange(samples_per_edge) / samples_per_edge
    edges = []
    for start, stop in zip(corners, np.roll(corners, -1, axis=0)):
        edges.append(
            start + fraction[:, None] * (stop - start)
        )
    return np.vstack([*edges, corners[0]])


def detector_footprint(detector, *, samples_per_edge=1):
    ''' Return the nominal detector boundary in focal-plane mm coordinates

    Detector distortion is intentionally excluded because computing the
    distorted boundary requires its inverse transformation.
    '''
    if not isinstance(detector, Detector):
        raise TypeError('`detector` should be a Detector instance.')
    if isinstance(samples_per_edge, bool):
        raise TypeError('`samples_per_edge` should be a positive integer.')
    try:
        samples_per_edge = integer_index(samples_per_edge)
    except TypeError as error:
        raise TypeError(
            '`samples_per_edge` should be a positive integer.'
        ) from error
    if samples_per_edge <= 0:
        raise ValueError('`samples_per_edge` should be positive.')

    pixel_xy = _rectangle_boundary(detector.shape, samples_per_edge)
    angle = -np.deg2rad(float(detector.rotation))
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), +np.cos(angle)],
    ])
    scaled = pixel_xy * np.asarray(detector.pixel_scale)
    return np.asarray(detector.offset) + (rotation @ scaled.T).T


def _clip_footprint(xy, radius, samples_per_edge):
    ''' Clip a focal-plane footprint to a circular imaging region '''
    circle = Point(0.0, 0.0).buffer(
        radius,
        quad_segs=max(8, samples_per_edge),
    )
    clipped = Polygon(xy).intersection(circle)
    if clipped.is_empty or not isinstance(clipped, Polygon):
        return np.empty((0, 2), dtype=float)
    return np.asarray(clipped.exterior.coords)


def _inverse_projection(projection, xy, scale, pointing):
    ''' Convert nominal focal-plane coordinates into ICRS coordinates '''
    scaled = xy / scale
    angle = np.deg2rad(float(pointing.position_angle[0]))
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), +np.cos(angle)],
    ])
    plane = (rotation @ scaled.T).T
    radius = np.linalg.norm(plane, axis=1)

    if isinstance(projection, GnomonicProjection):
        separation = np.arctan(np.deg2rad(radius))
    elif isinstance(projection, EquidistantProjection):
        separation = np.deg2rad(radius)
    else:
        raise TypeError(
            'Unsupported Projection type for footprint conversion.')

    position_angle = np.arctan2(-plane[:, 0], plane[:, 1])
    center = SkyCoord(
        ra=float(pointing.ra[0]) * u.deg,
        dec=float(pointing.dec[0]) * u.deg,
        frame='icrs',
    )
    return center.directional_offset_by(
        position_angle * u.rad,
        separation * u.rad,
    )


def telescope_footprints(
        telescope, exposure, *, frame='icrs', samples_per_edge=16,
        limit=True):
    ''' Return nominal sky footprints for one selected exposure

    Optical and detector distortions are intentionally excluded. The returned
    tuple follows the detector order in ``telescope.detectors``.
    '''
    if not isinstance(telescope, Telescope):
        raise TypeError('`telescope` should be a Telescope instance.')
    if not isinstance(exposure, Exposure):
        raise TypeError('`exposure` should be an Exposure instance.')
    if len(exposure) != 1:
        raise ValueError(
            '`exposure` should contain exactly one selected exposure.')
    if frame not in ('icrs', 'galactic'):
        raise ValueError('`frame` should be either "icrs" or "galactic".')
    if not isinstance(limit, bool):
        raise TypeError('`limit` should be a boolean.')

    scale_factor = np.asarray(
        exposure.calibration.scale_factor(jnp.array([0]))
    )[0]
    scale = np.asarray(telescope.optics.plate_scale) * scale_factor
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError(
            'The effective plate scale should be finite and nonzero.')

    footprints = []
    for detector in telescope.detectors:
        focal_plane = detector_footprint(
            detector,
            samples_per_edge=samples_per_edge,
        )
        radius = telescope.optics.imaging_radius
        if limit and radius is not None:
            focal_plane = _clip_footprint(
                focal_plane,
                radius,
                samples_per_edge,
            )
        sky = _inverse_projection(
            telescope.optics.projection,
            focal_plane,
            scale,
            exposure.pointing,
        )
        footprints.append(sky.transform_to(frame))
    return tuple(footprints)
