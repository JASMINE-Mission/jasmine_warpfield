#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Exposure-dependent astrometric parameters '''

import astropy.units as u
import numpy as np
import zodiax as zdx

from .calibration import Calibration, IdentityCalibration, ScaleCalibration
from .pointing import Pointing


__all__ = ['Exposure']


class Exposure(zdx.Base):
    ''' Composition of pointing and calibration parameters

    Attributes:
        pointing: Telescope attitude for each exposure.
        calibration: Exposure-dependent telescope calibration.
    '''

    pointing: Pointing
    calibration: Calibration

    def __init__(self, pointing, calibration=None):
        if calibration is None:
            calibration = IdentityCalibration()
        if not isinstance(pointing, Pointing):
            raise TypeError('`pointing` should be a Pointing instance.')
        if not isinstance(calibration, Calibration):
            raise TypeError(
                '`calibration` should be a Calibration instance.')
        if calibration.num_exposure not in (None, len(pointing)):
            raise ValueError(
                '`pointing` and `calibration` should have the same length.')

        self.pointing = pointing
        self.calibration = calibration

    def __len__(self):
        return len(self.pointing)

    def __getitem__(self, index):
        ''' Select exposures while preserving their PyTree structure '''
        return Exposure(
            self.pointing[index],
            self.calibration[index],
        )

    def __iter__(self):
        ''' Iterate over single-exposure collections '''
        for index in range(len(self)):
            yield self[index]

    def to_qtable(self):
        ''' Convert exposure parameters into a unit-aware QTable '''
        table = self.pointing.to_qtable()
        if isinstance(self.calibration, IdentityCalibration):
            table.meta['calibration'] = 'identity'
        elif isinstance(self.calibration, ScaleCalibration):
            table.meta['calibration'] = 'scale'
            table['scale_coefficient'] = (
                np.asarray(self.calibration.coefficient)
                * u.dimensionless_unscaled)
        else:
            raise TypeError(
                'Unsupported Calibration type for QTable conversion.')
        return table

    @classmethod
    def from_qtable(cls, table):
        ''' Construct exposure parameters from a unit-aware QTable '''
        pointing = Pointing.from_qtable(table)
        kind = table.meta.get('calibration')
        if kind is None:
            kind = 'scale' if 'scale_coefficient' in table.colnames \
                else 'identity'

        if kind == 'identity':
            calibration = IdentityCalibration()
        elif kind == 'scale':
            if 'scale_coefficient' not in table.colnames:
                raise ValueError(
                    '`table` is missing required column: scale_coefficient')
            coefficient = u.Quantity(
                table['scale_coefficient']).to_value(
                    u.dimensionless_unscaled)
            calibration = ScaleCalibration(coefficient)
        else:
            raise ValueError(f'Unsupported calibration type: {kind}')
        return cls(pointing, calibration)

    def take(self, index):
        ''' Select pointing and calibration parameters by exposure index '''
        return (
            *self.pointing.take(index),
            self.calibration.scale_factor(index),
        )
