#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Exposure-dependent astrometric parameters '''

import zodiax as zdx

from .calibration import Calibration
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

    def __init__(self, pointing, calibration):
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

    def take(self, index):
        ''' Select pointing and calibration parameters by exposure index '''
        return (
            *self.pointing.take(index),
            self.calibration.scale_factor(index),
        )
