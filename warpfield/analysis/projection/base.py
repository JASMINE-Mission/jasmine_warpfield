#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Base class for differentiable projection models '''

import zodiax as zdx


__all__ = ['Projection']


class Projection(zdx.Base):
    ''' Interface for projections from sky to focal-plane coordinates '''

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        raise NotImplementedError
