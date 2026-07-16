#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Nominal detector footprints in focal-plane and celestial coordinates '''

from operator import index as integer_index

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from shapely.geometry import Point, Polygon

from .detector import Detector
from .exposure import Exposure
from .pointing import Pointing
from .projection import EquidistantProjection, GnomonicProjection
from .telescope import Telescope


__all__ = ['celestial_footprints', 'focalplane_footprints']


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


def _validate_samples_per_edge(samples_per_edge):
    ''' Validate and normalize the number of boundary samples '''
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
    return samples_per_edge


def _detectors_from(detectors):
    ''' Normalize supported detector containers to a detector tuple '''
    if isinstance(detectors, Telescope):
        return detectors.detectors
    if isinstance(detectors, Detector):
        return (detectors,)
    if (
            isinstance(detectors, tuple)
            and all(isinstance(detector, Detector) for detector in detectors)):
        return detectors
    raise TypeError(
        '`detectors` should be a Telescope, Detector, or tuple of Detector.')


def _focalplane_footprint(detector, samples_per_edge):
    ''' Return one nominal detector boundary in focal-plane coordinates '''
    pixel_xy = _rectangle_boundary(detector.shape, samples_per_edge)
    angle = -np.deg2rad(float(detector.rotation))
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), +np.cos(angle)],
    ])
    scaled = pixel_xy * np.asarray(detector.pixel_scale)
    return np.asarray(detector.offset) + (rotation @ scaled.T).T


def focalplane_footprints(detectors, *, samples_per_edge=1):
    ''' Return nominal detector boundaries in focal-plane mm coordinates

    Detector distortion is intentionally excluded because computing the
    distorted boundaries requires its inverse transformation. The returned
    tuple follows the input detector order.
    '''
    detectors = _detectors_from(detectors)
    samples_per_edge = _validate_samples_per_edge(samples_per_edge)
    return tuple(
        _focalplane_footprint(detector, samples_per_edge)
        for detector in detectors
    )


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


def celestial_footprints(
        telescope, pointing, *, frame='icrs', samples_per_edge=16,
        limit=True):
    ''' Return nominal sky footprints for one selected pointing

    Optical and detector distortions are intentionally excluded. The returned
    tuple follows the detector order in ``telescope.detectors``.
    '''
    if not isinstance(telescope, Telescope):
        raise TypeError('`telescope` should be a Telescope instance.')
    if isinstance(pointing, Exposure):
        pointing = pointing.pointing
    elif not isinstance(pointing, Pointing):
        raise TypeError(
            '`pointing` should be a Pointing or Exposure instance.')
    if len(pointing) != 1:
        raise ValueError(
            '`pointing` should contain exactly one selected pointing.')
    if frame not in ('icrs', 'galactic'):
        raise ValueError('`frame` should be either "icrs" or "galactic".')
    if not isinstance(limit, bool):
        raise TypeError('`limit` should be a boolean.')

    samples_per_edge = _validate_samples_per_edge(samples_per_edge)
    scale = np.asarray(telescope.optics.plate_scale)
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError(
            'The plate scale should be finite and nonzero.')

    footprints = []
    focalplane = focalplane_footprints(
        telescope,
        samples_per_edge=samples_per_edge,
    )
    for focal_plane in focalplane:
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
            pointing,
        )
        footprints.append(sky.transform_to(frame))
    return tuple(footprints)
