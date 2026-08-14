#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""External astrometric catalog queries and adapters"""

from .gaia import compile_from_gaia, query_gaia


__all__ = [
    'compile_from_gaia',
    'query_gaia',
]
