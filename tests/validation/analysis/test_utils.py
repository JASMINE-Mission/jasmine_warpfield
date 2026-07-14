#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hypothesis import given
from hypothesis.strategies import floats
import numpy as np
from pytest import approx

from warpfield.analysis.utils import _degree_to_radian


def degree():
    return floats(0.0, 360.0)


@given(degree())
def test_degree_to_radian(theta):
    assert _degree_to_radian(theta) == approx(theta * np.pi / 180)
