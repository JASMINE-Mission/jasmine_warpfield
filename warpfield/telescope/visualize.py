#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Providing functions for visualization '''

import matplotlib.pyplot as plt

from .util import get_axis_name, get_projection
from .util import frame_conversion, estimate_frame_from_ctype
from .source import SourceTable


def get_subplot(pointing, key=111, figsize=(8, 8)):
    ''' Generate an axis instance for a poiting

    Arguments:
      pointing (SkyCoord):
          The directin of the telescope pointing.
      frame (string):
          Set to override the projection of `pointing`.
    '''
    proj = get_projection(pointing)

    fig = plt.figure(figsize=figsize)
    axis = fig.add_subplot(key, projection=proj)

    return fig, axis


def display_sources(axis, sources, **options):
    ''' Display sources around the specified coordinates

    Arguments:
      axis (Axes):
          Matplotlib Axes instance.
      sources (SkyCoord or SourceTable):
          The list of sources.
    '''
    if isinstance(sources, SourceTable):
        sources = sources.skycoord
    frame = options.get('frame')
    if frame is None:
        ctype = axis.wcs.wcs.ctype
        frame = estimate_frame_from_ctype(ctype)

    skycoord = frame_conversion(sources, frame)
    xlabel, ylabel = get_axis_name(frame)

    title = options.pop('title', None)
    marker = options.pop('marker', 'x')
    axis.set_aspect(1.0)
    axis.set_position([0.13, 0.10, 0.85, 0.85])
    axis.scatter(skycoord.spherical.lon,
                 skycoord.spherical.lat,
                 transform=axis.get_transform(frame),
                 marker=marker,
                 label=title,
                 **options)
    axis.grid()
    if title is not None:
        axis.legend(bbox_to_anchor=[1, 1], loc='lower right', frameon=False)
    axis.set_xlabel(xlabel, fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
