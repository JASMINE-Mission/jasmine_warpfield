#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base class for differentiable distortion models"""

import zodiax as zdx


__all__ = ['Distortion']


class Distortion(zdx.Base):
    """Interface for models that return coordinate displacements"""

    def __call__(self, xy):
        raise NotImplementedError
