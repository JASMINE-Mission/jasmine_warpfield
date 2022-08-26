#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, assume, settings
from hypothesis.strategies import integers
import numpy as np

from warpfield.telescope.distortion.sip import *


@given(integers(0, 2**31))
def test_sip_distortion_zero(seed):
    Nsrc = 1000
    order = 1
    A = np.zeros((order + 1, order + 1))
    B = np.zeros((order + 1, order + 1))
    np.random.seed(seed)
    position = np.random.uniform(-1000, 1000, size=(2, Nsrc))

    sip = SipDistortion(order, A, B)
    converted = sip(position)
    assert converted == approx(position)

@given(integers(0, 2**31))
def test_altsip_distortion_zero(seed):
    Nsrc = 1000
    order = 1
    A = np.zeros((order + 1, order + 1))
    B = np.zeros((order + 1, order + 1))
    np.random.seed(seed)
    c = np.random.uniform(-10,10, size=(2, 1))
    position = np.random.uniform(-1000, 1000, size=(2, Nsrc))

    sip = AltSipDistortion(order, c, A, B)
    converted = sip(position)
    assert converted == approx(position)
