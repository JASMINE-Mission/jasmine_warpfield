#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
import numpy as np

from warpfield.telescope.distortion.identity import *


def test_identity(seed=42, Nsrc=1000):
    np.random.seed(seed)
    position = np.random.normal(size=(2, Nsrc))
    converted = identity_transformation(position)
    assert converted == approx(position)
