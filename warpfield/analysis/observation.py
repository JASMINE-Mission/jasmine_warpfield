#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Astrometric observations for differentiable analysis '''

from jax import Array
import jax.numpy as jnp
import zodiax as zdx


__all__ = ['Observation']


class Observation(zdx.Base):
    ''' Detector measurements and their model associations

    Attributes:
        xy: Measured detector coordinates with shape
            ``(N_observation, 2)``.
        source_index: Source catalog indices with shape
            ``(N_observation,)``.
        pointing_index: Telescope pointing indices with shape
            ``(N_observation,)``.
        detector_index: Detector indices with shape
            ``(N_observation,)``.
        uncertainty: Optional coordinate uncertainties with shape
            ``(N_observation, 2)``.
    '''

    xy: Array
    source_index: Array
    pointing_index: Array
    detector_index: Array
    uncertainty: Array | None

    def __init__(
            self, xy, source_index, pointing_index, detector_index,
            uncertainty=None):
        xy = jnp.asarray(xy, dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_observation, 2).')

        size = xy.shape[0]
        source_index = self._validate_index(
            source_index, 'source_index', size)
        pointing_index = self._validate_index(
            pointing_index, 'pointing_index', size)
        detector_index = self._validate_index(
            detector_index, 'detector_index', size)

        if uncertainty is not None:
            uncertainty = jnp.asarray(uncertainty, dtype=float)
            if uncertainty.shape != xy.shape:
                raise ValueError(
                    '`uncertainty` should have shape '
                    '(N_observation, 2).')

        self.xy = xy
        self.source_index = source_index
        self.pointing_index = pointing_index
        self.detector_index = detector_index
        self.uncertainty = uncertainty

    @staticmethod
    def _validate_index(index, name, size):
        index = jnp.asarray(index)

        if index.ndim != 1:
            raise ValueError(f'`{name}` should be a one-dimensional array.')
        if index.shape[0] != size:
            raise ValueError(
                f'`{name}` should have length N_observation.')
        if not jnp.issubdtype(index.dtype, jnp.integer):
            raise ValueError(f'`{name}` should contain integers.')

        return index

    def __len__(self):
        return self.xy.shape[0]
