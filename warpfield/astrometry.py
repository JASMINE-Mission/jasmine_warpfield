#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Differentiable astrometric coordinate prediction"""

import zodiax as zdx

from .exposure import Exposure
from .measurement import Measurement
from .catalog import SourceCatalog
from .telescope import Telescope


__all__ = ['Astrometry']


class Astrometry(zdx.Base):
    """Composition of source, telescope, and exposure parameters

    Attributes:
        source: Celestial source parameters.
        telescope: Telescope parameters and coordinate transformations.
        exposure: Pointing and calibration parameters for each exposure.
    """

    source: SourceCatalog
    telescope: Telescope
    exposure: Exposure

    def __init__(self, source, telescope, exposure):
        if not isinstance(source, SourceCatalog):
            raise TypeError('`source` should be a SourceCatalog instance.')
        if not isinstance(telescope, Telescope):
            raise TypeError('`telescope` should be a Telescope instance.')
        if not isinstance(exposure, Exposure):
            raise TypeError('`exposure` should be an Exposure instance.')

        self.source = source
        self.telescope = telescope
        self.exposure = exposure

    @staticmethod
    def _validate_measurement(measurement):
        if not isinstance(measurement, Measurement):
            raise TypeError('`measurement` should be a Measurement instance.')
        return measurement

    def __call__(self, measurement):
        """Predict detector coordinates for the measurements"""
        measurement = self._validate_measurement(measurement)
        ra, dec = self.source.take(measurement.source_index)
        tel_ra, tel_dec, tel_pa, scale_factor = self.exposure.take(
            measurement.exposure_index
        )
        return self.telescope(
            tel_ra,
            tel_dec,
            tel_pa,
            ra,
            dec,
            scale_factor,
            measurement.detector_index,
        )

    def residual(self, measurement):
        """Return observed minus predicted detector coordinates"""
        measurement = self._validate_measurement(measurement)
        return measurement.xy - self(measurement)
