#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx

from .util import *
from warpfield.analysis.distortion.legendre import *


def test_2d_legendre(x, random):
    c = random.normal(size=(10, 10))
    legval2d(x, x, c)


def test_distortion(xy, random):
    coeff_a = random.normal(size=(18))
    coeff_b = random.normal(size=(18))
    d = distortion(coeff_a, coeff_b, xy)
    assert at_origin(d) == approx(0.0)
