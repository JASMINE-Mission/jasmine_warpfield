#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Astrometric measurements for differentiable analysis"""

from astropy.table import QTable
import astropy.units as u
from jax import Array
import jax.numpy as jnp
import numpy as np
import zodiax as zdx


__all__ = ['Measurement']


class Measurement(zdx.Base):
    """Detector measurements and their parameter associations

    Attributes:
        xy: Measured detector coordinates with shape
            ``(N_measurement, 2)``.
        source_index: Source catalog indices with shape
            ``(N_measurement,)``.
        exposure_index: Exposure indices with shape ``(N_measurement,)``.
        detector_index: Detector indices with shape ``(N_measurement,)``.
        uncertainty: Optional coordinate uncertainties with shape
            ``(N_measurement, 2)``.
    """

    xy: Array
    source_index: Array
    exposure_index: Array
    detector_index: Array
    uncertainty: Array | None

    def __init__(
        self,
        xy,
        source_index,
        exposure_index,
        detector_index,
        uncertainty=None,
    ):
        xy = jnp.asarray(xy, dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_measurement, 2).')

        size = xy.shape[0]
        source_index = self._validate_index(source_index, 'source_index', size)
        exposure_index = self._validate_index(
            exposure_index, 'exposure_index', size
        )
        detector_index = self._validate_index(
            detector_index, 'detector_index', size
        )

        if uncertainty is not None:
            uncertainty = jnp.asarray(uncertainty, dtype=float)
            if uncertainty.shape != xy.shape:
                raise ValueError(
                    '`uncertainty` should have shape (N_measurement, 2).'
                )

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

    def to_qtable(self):
        """Convert measurements into a unit-aware QTable"""
        table = QTable({
            'measurement_id': np.arange(len(self), dtype=int),
            'x': np.asarray(self.xy[:, 0]) * u.pix,
            'y': np.asarray(self.xy[:, 1]) * u.pix,
            'source_id': np.asarray(self.source_index, dtype=int),
            'exposure_id': np.asarray(self.exposure_index, dtype=int),
            'detector_id': np.asarray(self.detector_index, dtype=int),
        })
        if self.uncertainty is not None:
            table['x_error'] = np.asarray(self.uncertainty[:, 0]) * u.pix
            table['y_error'] = np.asarray(self.uncertainty[:, 1]) * u.pix
        return table

    @classmethod
    def from_qtable(cls, table):
        """Construct measurements from a unit-aware QTable"""
        if not isinstance(table, QTable):
            raise TypeError('`table` should be a QTable instance.')
        required = ('x', 'y', 'source_id', 'exposure_id', 'detector_id')
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise ValueError(
                '`table` is missing required columns: ' + ', '.join(missing)
            )
        try:
            xy = np.stack(
                [
                    u.Quantity(table[name]).to_value(u.pix)
                    for name in ('x', 'y')
                ],
                axis=1,
            )
        except u.UnitConversionError as error:
            raise ValueError(
                'Measurement coordinates should have pixel units.'
            ) from error

        error_columns = [
            name in table.colnames for name in ('x_error', 'y_error')
        ]
        if any(error_columns) and not all(error_columns):
            raise ValueError(
                '`table` should contain both x_error and y_error.'
            )
        uncertainty = None
        if all(error_columns):
            try:
                uncertainty = np.stack(
                    [
                        u.Quantity(table[name]).to_value(u.pix)
                        for name in ('x_error', 'y_error')
                    ],
                    axis=1,
                )
            except u.UnitConversionError as error:
                raise ValueError(
                    'Measurement errors should have pixel units.'
                ) from error

        return cls(
            xy=xy,
            source_index=table['source_id'],
            exposure_index=table['exposure_id'],
            detector_index=table['detector_id'],
            uncertainty=uncertainty,
        )
