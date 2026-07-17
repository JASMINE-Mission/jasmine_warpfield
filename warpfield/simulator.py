#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Telescope simulator with focal-plane and detector masks '''

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from operator import index as integer_index

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .detector import Detector, _focal_plane_corners
from .exposure import Exposure
from .measurement import Measurement
from .source import SourceCatalog
from .telescope import Telescope


__all__ = ['ErrorGenerator', 'Simulator', 'UniformError']


class ErrorGenerator(ABC):
    ''' Interface for simulated measurement errors and uncertainties '''

    @abstractmethod
    def __call__(self, source, seed, shape):
        ''' Generate one error realization with the requested shape '''
        raise NotImplementedError

    @abstractmethod
    def uncertainty(self, source, shape):
        ''' Return standard uncertainties with the requested shape '''
        raise NotImplementedError


@dataclass(frozen=True)
class UniformError(ErrorGenerator):
    ''' Homoscedastic normally distributed measurement errors '''

    standard_deviation: float

    def __post_init__(self):
        if (
                isinstance(self.standard_deviation, bool)
                or not isinstance(self.standard_deviation, Real)):
            raise TypeError(
                '`standard_deviation` should be a non-negative float.')
        standard_deviation = float(self.standard_deviation)
        if (
                not np.isfinite(standard_deviation)
                or standard_deviation < 0):
            raise ValueError(
                '`standard_deviation` should be finite and non-negative.')
        object.__setattr__(
            self,
            'standard_deviation',
            standard_deviation,
        )

    def __call__(self, source, seed, shape):
        ''' Generate independent normally distributed errors '''
        del source
        return np.random.default_rng(seed).normal(
            loc=0.0,
            scale=self.standard_deviation,
            size=shape,
        )

    def uncertainty(self, source, shape):
        ''' Return the common Gaussian standard deviation '''
        del source
        return np.full(shape, self.standard_deviation, dtype=float)


def _validate_seed(seed):
    ''' Validate and normalize a random-number seed '''
    if seed is None:
        return None
    if isinstance(seed, bool):
        raise TypeError('`seed` should be a non-negative integer or None.')
    try:
        seed = integer_index(seed)
    except TypeError as error:
        raise TypeError(
            '`seed` should be a non-negative integer or None.'
        ) from error
    if seed < 0:
        raise ValueError('`seed` should be non-negative.')
    return seed


def _normalize_error(error):
    ''' Normalize an error specification to an ErrorGenerator '''
    if error is None:
        return None
    if isinstance(error, bool):
        raise TypeError(
            '`error` should be a non-negative float, '
            'ErrorGenerator, or None.')
    if isinstance(error, Real):
        try:
            return UniformError(error)
        except (TypeError, ValueError) as exception:
            raise type(exception)(
                '`error` should be finite and non-negative.'
            ) from exception
    if isinstance(error, ErrorGenerator):
        return error
    raise TypeError(
        '`error` should be a non-negative float, ErrorGenerator, or None.')


def _generate_errors(generator, source, seed, shape):
    ''' Generate and validate measurement errors and uncertainties '''
    error = np.asarray(generator(source, seed, shape), dtype=float)
    uncertainty = np.asarray(
        generator.uncertainty(source, shape),
        dtype=float,
    )
    if error.shape != shape:
        raise ValueError(
            'Generated errors should have shape '
            f'{shape}, but received {error.shape}.')
    if uncertainty.shape != shape:
        raise ValueError(
            'Generated uncertainties should have shape '
            f'{shape}, but received {uncertainty.shape}.')
    if not np.all(np.isfinite(error)):
        raise ValueError('Generated errors should be finite.')
    if (
            not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty < 0)):
        raise ValueError(
            'Generated uncertainties should be finite and non-negative.')
    return error, uncertainty


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

    def observe(self, source, exposure, *, error=None, seed=0):
        ''' Generate measurements inside the configured masks

        ``error`` may be a non-negative Gaussian standard deviation in pixels
        or an ErrorGenerator. Generated errors are added to ideal detector
        coordinates, and their standard uncertainties are stored in the
        returned Measurement.
        '''
        if not isinstance(source, SourceCatalog):
            raise TypeError('`source` should be a SourceCatalog instance.')
        if not isinstance(exposure, Exposure):
            raise TypeError('`exposure` should be an Exposure instance.')
        generator = _normalize_error(error)
        seed = _validate_seed(seed)

        exposure_index, source_index = jnp.meshgrid(
            jnp.arange(len(exposure)),
            jnp.arange(len(source)),
            indexing='ij',
        )
        exposure_index = exposure_index.ravel()
        source_index = source_index.ravel()

        ra, dec = source.take(source_index)
        tel_ra, tel_dec, tel_pa, scale_factor = \
            exposure.take(exposure_index)
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

        coordinates = jnp.concatenate(coordinates, axis=0)
        source_indices = jnp.concatenate(source_indices)
        exposure_indices = jnp.concatenate(exposure_indices)
        detector_indices = jnp.concatenate(detector_indices)
        uncertainty = None
        if generator is not None:
            generated, uncertainty = _generate_errors(
                generator,
                source[source_indices],
                seed,
                coordinates.shape,
            )
            coordinates = coordinates + generated

        return Measurement(
            xy=coordinates,
            source_index=source_indices,
            exposure_index=exposure_indices,
            detector_index=detector_indices,
            uncertainty=uncertainty,
        )
