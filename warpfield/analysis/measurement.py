#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Astrometric measurements for differentiable analysis '''

from jax import Array
import jax.numpy as jnp
import zodiax as zdx


__all__ = ['Measurement']


class Measurement(zdx.Base):
    ''' Detector measurements and their parameter associations

    Attributes:
        xy: Measured detector coordinates with shape
            ``(N_measurement, 2)``.
        source_index: Source catalog indices with shape
            ``(N_measurement,)``.
        exposure_index: Exposure indices with shape ``(N_measurement,)``.
        detector_index: Detector indices with shape ``(N_measurement,)``.
        uncertainty: Optional coordinate uncertainties with shape
            ``(N_measurement, 2)``.
    '''

    xy: Array
    source_index: Array
    exposure_index: Array
    detector_index: Array
    uncertainty: Array | None

    def __init__(
            self, xy, source_index, exposure_index, detector_index,
            uncertainty=None):
        xy = jnp.asarray(xy, dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_measurement, 2).')

        size = xy.shape[0]
        source_index = self._validate_index(
            source_index, 'source_index', size)
        exposure_index = self._validate_index(
            exposure_index, 'exposure_index', size)
        detector_index = self._validate_index(
            detector_index, 'detector_index', size)

        if uncertainty is not None:
            uncertainty = jnp.asarray(uncertainty, dtype=float)
            if uncertainty.shape != xy.shape:
                raise ValueError(
                    '`uncertainty` should have shape (N_measurement, 2).')

        self.xy = xy
        self.source_index = source_index
        self.exposure_index = exposure_index
        self.detector_index = detector_index
        self.uncertainty = uncertainty

    @staticmethod
    def _validate_index(index, name, size):
        index = jnp.asarray(index)

        if index.ndim != 1:
            raise ValueError(f'`{name}` should be a one-dimensional array.')
        if index.shape[0] != size:
            raise ValueError(f'`{name}` should have length N_measurement.')
        if not jnp.issubdtype(index.dtype, jnp.integer):
            raise ValueError(f'`{name}` should contain integers.')

        return index

    def __len__(self):
        return self.xy.shape[0]
