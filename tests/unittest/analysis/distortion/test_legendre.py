#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
import numpy.polynomial.legendre as legendre


from .util import *
from warpfield.analysis.distortion.legendre import *


def test_1d_legendre(y, random):
    c = random.normal(size=(10))
    assert legval(y, c) == approx(legendre.legval(y, c))


def test_2d_legendre(x, y, random):
    c = random.normal(size=(10, 10))
    assert legval2d(x, y, c) == approx(legendre.legval2d(x, y, c))


def test_distortion(xy, random):
    coeff_a = random.normal(size=(18))
    coeff_b = random.normal(size=(18))
    d = distortion(coeff_a, coeff_b, xy)
    assert at_origin(d) == approx(0.0)
