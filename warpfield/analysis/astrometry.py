#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable astrometric coordinate prediction '''

import zodiax as zdx

from .observation import Observation
from .source import SourceCatalog
from .telescope import Telescope


__all__ = ['Astrometry']


class Astrometry(zdx.Base):
    ''' Composition of a source catalog and telescope model

    Attributes:
        source: Celestial source parameters.
        telescope: Telescope parameters and coordinate transformations.
    '''

    source: SourceCatalog
    telescope: Telescope

    def __init__(self, source, telescope):
        if not isinstance(source, SourceCatalog):
            raise TypeError('`source` should be a SourceCatalog instance.')
        if not isinstance(telescope, Telescope):
            raise TypeError('`telescope` should be a Telescope instance.')

        self.source = source
        self.telescope = telescope

    @staticmethod
    def _validate_observation(observation):
        if not isinstance(observation, Observation):
            raise TypeError(
                '`observation` should be an Observation instance.')
        return observation

    def __call__(self, observation):
        ''' Predict detector coordinates for the observations '''
        observation = self._validate_observation(observation)
        ra, dec = self.source.take(observation.source_index)
        return self.telescope(
            ra,
            dec,
            observation.pointing_index,
            observation.detector_index,
        )

    def residual(self, observation):
        ''' Return observed minus predicted detector coordinates '''
        observation = self._validate_observation(observation)
        return observation.xy - self(observation)
