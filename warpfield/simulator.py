#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Telescope simulator with focal-plane and detector masks '''

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp

from .detector import Detector, _focal_plane_corners
from .exposure import Exposure
from .measurement import Measurement
from .source import SourceCatalog
from .telescope import Telescope


__all__ = ['Simulator']


@dataclass(frozen=True)
class _CircularMask:
    radius: float

    def __call__(self, xy):
        xy = jnp.asarray(xy)
        radius_squared = jnp.asarray(self.radius, dtype=xy.dtype)**2
        tolerance = (
            4 * jnp.finfo(xy.dtype).eps
            * jnp.maximum(1, radius_squared)
        )
        return jnp.sum(xy**2, axis=1) <= radius_squared + tolerance


@dataclass(frozen=True)
class _DetectorMask:
    shape: tuple[int, int]

    def __call__(self, xy):
        xy = jnp.asarray(xy)
        limit = jnp.asarray(self.shape, dtype=xy.dtype) / 2
        return jnp.all((0 <= xy) & (xy <= 2 * limit), axis=1)


def _generate_detector_mask(detector):
    ''' Generate a detector mask and its focal-plane corner coordinates '''
    if not isinstance(detector, Detector):
        raise TypeError('`detector` should be a Detector instance.')

    mask = _DetectorMask(detector.shape)
    return mask, _focal_plane_corners(detector)


def _generate_masks(optics, detectors):
    ''' Generate detector masks and their enclosing focal-plane mask '''
    detector_masks = []
    for detector in detectors:
        mask, _ = _generate_detector_mask(detector)
        detector_masks.append(mask)
    return (
        _CircularMask(optics.imaging_radius),
        tuple(detector_masks),
    )


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

    def __init__(
            self, projection, plate_scale, detectors, *,
            distortion=None, imaging_radius=None):
        super().__init__(
            projection,
            plate_scale,
            detectors,
            distortion=distortion,
            imaging_radius=imaging_radius,
        )
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
