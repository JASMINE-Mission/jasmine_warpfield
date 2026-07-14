#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx

from .util import *
from warpfield.analysis.distortion.sip import *


def test_polymap(xy):
    coeff = jnp.array([0.0, 0.0, 0.4])
    polymap(coeff, xy)


def test_distortion(xy, random):
    coeff_a = random.normal(size=(18))
    coeff_b = random.normal(size=(18))
    d = distortion(coeff_a, coeff_b, xy)
    assert at_origin(d) == approx(0.0)

    d = distortion(0 * coeff_a, 0 * coeff_b, xy)
    assert d == approx(jnp.zeros_like(xy))

    d = distortion(0 * coeff_a, coeff_b, xy)
    assert d[:, 0] == approx(0.0)

    d = distortion(coeff_a, 0 * coeff_b, xy)
    assert d[:, 1] == approx(0.0)
