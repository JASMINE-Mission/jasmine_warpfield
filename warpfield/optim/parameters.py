#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Zodiax parameter selection helpers"""

from collections.abc import Mapping

import zodiax as zdx


__all__ = ['get_parameters', 'set_parameters']


def get_parameters(tree, paths):
    """Extract selected PyTree leaves as a path-keyed dictionary"""
    if not isinstance(tree, zdx.Base):
        raise TypeError('`tree` should be a zodiax Base instance.')
    if isinstance(paths, str):
        paths = [paths]
    else:
        paths = list(paths)
    if len(paths) == 0:
        raise ValueError('`paths` should contain at least one path.')
    return tree.get(paths, as_dict=True, to_array=False)


def set_parameters(tree, parameters):
    """Return a PyTree with the selected leaves replaced"""
    if not isinstance(tree, zdx.Base):
        raise TypeError('`tree` should be a zodiax Base instance.')
    if not isinstance(parameters, Mapping):
        raise TypeError('`parameters` should be a mapping.')
    if len(parameters) == 0:
        raise ValueError('`parameters` should contain at least one value.')
    return tree.set(dict(parameters))
