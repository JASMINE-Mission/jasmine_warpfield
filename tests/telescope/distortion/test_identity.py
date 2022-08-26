#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, assume, settings
from hypothesis.strategies import integers
import numpy as np

from warpfield.telescope.distortion.identity import *

@settings(deadline=500, max_examples=10)
@given(integers(1, 10000))
def test_identity(Nsrc):
    position = np.zeros((2, Nsrc))
    converted = identity_transformation(position)
    assert converted == approx(position)
