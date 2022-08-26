#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given
from hypothesis import assume
from hypothesis.strategies import floats
import numpy as np

from warpfield.analysis.projection.gnomonic import *

@given(floats(0,360))
def test_gnomonic(theta):
    rad = degree_to_radian(theta)
    assert rad == approx(theta * np.pi/180)

def test_gnomonic_x_zero():
    x = gnomonic_x(0.0, 0.0, 0.0, 0.0)
    assert x == approx(0.0)
