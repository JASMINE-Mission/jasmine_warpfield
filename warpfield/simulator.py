#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Telescope simulator with focal-plane and detector masks '''

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .detector import Detector
from .exposure import Exposure
from .measurement import Measurement
from .optics import Optics
from .source import SourceCatalog
from .telescope import Telescope


__all__ = ['Simulator']


@dataclass(frozen=True)
class _CircularMask:
    radius: float

    def __call__(self, xy):
        xy = jnp.asarray(xy)
        return jnp.sum(xy**2, axis=1) <= self.radius**2


@dataclass(frozen=True)
class _DetectorMask:
    shape: tuple[int, int]

    def __call__(self, xy):
        xy = jnp.asarray(xy)
        limit = jnp.asarray(self.shape, dtype=xy.dtype) / 2
        return jnp.all((-limit <= xy) & (xy <= limit), axis=1)


def _generate_detector_mask(detector):
    ''' Generate a detector mask and its focal-plane corner coordinates '''
    if not isinstance(detector, Detector):
        raise TypeError('`detector` should be a Detector instance.')

    mask = _DetectorMask(detector.shape)
    half = np.asarray(detector.shape, dtype=float) / 2
    corners = np.array([
        [-half[0], -half[1]],
        [+half[0], -half[1]],
        [+half[0], +half[1]],
        [-half[0], +half[1]],
    ])

    angle = -np.deg2rad(float(detector.rotation))
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), +np.cos(angle)],
    ])
    scaled = corners * np.asarray(detector.pixel_scale)
    focal_plane = np.asarray(detector.offset) + (rotation @ scaled.T).T
    return mask, focal_plane


def _generate_masks(optics, detectors):
    ''' Generate detector masks and their enclosing focal-plane mask '''
    if not isinstance(optics, Optics):
        raise TypeError('`optics` should be an Optics instance.')

    detector_masks = []
    focal_plane_corners = []
    for detector in detectors:
        mask, corners = _generate_detector_mask(detector)
        detector_masks.append(mask)
        focal_plane_corners.append(corners)

    corners = np.concatenate(focal_plane_corners)
    radius = float(np.sqrt(np.sum(corners**2, axis=1)).max())
    if optics.imaging_radius is not None:
        radius = optics.imaging_radius
    return _CircularMask(radius), tuple(detector_masks)


class Simulator(Telescope):
    ''' Telescope variant that generates masked ideal measurements

    Attributes:
        optics: Projection and focal-plane distortion model.
        detectors: Detector geometry models.
        fov_mask: Circular focal-plane mask enclosing all detectors.
        detector_masks: Rectangular masks in detector coordinates.
    '''

    fov_mask: _CircularMask = eqx.field(static=True)
    detector_masks: tuple[_DetectorMask, ...] = eqx.field(static=True)

    def __init__(self, optics, detectors):
        super().__init__(optics, detectors)
        fov_mask, detector_masks = _generate_masks(
            self.optics,
            self.detectors,
        )

        self.fov_mask = fov_mask
        self.detector_masks = detector_masks

    @property
    def fov_radius(self):
        ''' Radius of the circular focal-plane mask in mm '''
        return self.fov_mask.radius

    def observe(self, source, exposure):
        ''' Generate ideal measurements inside the configured masks '''
        if not isinstance(source, SourceCatalog):
            raise TypeError('`source` should be a SourceCatalog instance.')
        if not isinstance(exposure, Exposure):
            raise TypeError('`exposure` should be an Exposure instance.')

        exposure_index, source_index = jnp.meshgrid(
            jnp.arange(len(exposure)),
            jnp.arange(len(source)),
            indexing='ij',
        )
        exposure_index = exposure_index.ravel()
        source_index = source_index.ravel()

        ra, dec = source.take(source_index)
        tel_ra, tel_dec, tel_pa, scale_factor = exposure.take(
            exposure_index)
        focal_plane = self.focal_plane(
            tel_ra,
            tel_dec,
            tel_pa,
            ra,
            dec,
            scale_factor,
        )

        within_fov = self.fov_mask(focal_plane)
        focal_plane = focal_plane[within_fov]
        source_index = source_index[within_fov]
        exposure_index = exposure_index[within_fov]

        coordinates = []
        source_indices = []
        exposure_indices = []
        detector_indices = []
        for index, (detector, mask) in enumerate(zip(
                self.detectors, self.detector_masks)):
            detector_xy = detector(focal_plane)
            within_detector = mask(detector_xy)
            size = int(jnp.sum(within_detector))

            coordinates.append(detector_xy[within_detector])
            source_indices.append(source_index[within_detector])
            exposure_indices.append(exposure_index[within_detector])
            detector_indices.append(jnp.full(size, index, dtype=int))

        return Measurement(
            xy=jnp.concatenate(coordinates, axis=0),
            source_index=jnp.concatenate(source_indices),
            exposure_index=jnp.concatenate(exposure_indices),
            detector_index=jnp.concatenate(detector_indices),
        )
