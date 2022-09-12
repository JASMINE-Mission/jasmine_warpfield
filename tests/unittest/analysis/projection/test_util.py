#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
import numpy as np

from warpfield.analysis.projection.util import *


def test_degree_to_radian():
    args = [-180, -90, 0, 90, 180]
    for theta in args:
        rad = degree_to_radian(theta)
        assert rad == approx(theta * np.pi / 180)


def test_rotation_matrix():
    assert rotation_matrix(0.0).ravel() == approx([1, 0, 0, 1])
    assert rotation_matrix(np.pi / 2).ravel() == approx([0, -1, 1, 0])
