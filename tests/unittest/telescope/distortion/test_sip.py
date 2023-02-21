#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
import numpy as np

from warpfield.telescope.distortion.sip import *


def test_sip_distortion_zero(seed=42, Nsrc=1000):
    order = 1
    A = np.zeros((order + 1, order + 1))
    B = np.zeros((order + 1, order + 1))
    np.random.seed(seed)
    position = np.random.uniform(-1000, 1000, size=(2, Nsrc))

    func = SipDistortion(order, A, B)
    converted = func(position)
    assert converted == approx(position)


def test_altsip_distortion_zero(seed=42, Nsrc=1000):
    order = 1
    A = np.zeros((order + 1, order + 1))
    B = np.zeros((order + 1, order + 1))
    np.random.seed(seed)
    center = np.random.uniform(-10, 10, size=(2, 1))
    position = np.random.uniform(-1000, 1000, size=(2, Nsrc))

    func = DisplacedSipDistortion(order, center, A, B)
    converted = func(position)
    assert converted == approx(position)
