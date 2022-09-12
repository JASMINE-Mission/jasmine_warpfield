#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, assume, settings
from hypothesis.strategies import integers
import numpy as np

from warpfield.telescope import *
