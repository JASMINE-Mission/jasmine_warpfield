#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given
from hypothesis.strategies import floats
import numpy as np

from warpfield.analysis.conversion import *


def degree():
    return floats(0.0, 360.0)


@given(degree())
def test_degree_to_radian(theta):
    rad = degree_to_radian(theta)
    assert rad == approx(theta * np.pi / 180)
