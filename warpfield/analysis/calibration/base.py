#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Base class for exposure calibration '''

import zodiax as zdx


__all__ = ['Calibration']


class Calibration(zdx.Base):
    ''' Interface for exposure-dependent calibration parameters '''

    @property
    def num_exposure(self):
        ''' Number of calibrated exposures, or None when unrestricted '''
        return None

    def scale_factor(self, exposure_index):
        ''' Return plate-scale factors for the selected exposures '''
        raise NotImplementedError
