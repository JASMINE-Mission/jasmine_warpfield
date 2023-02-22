#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Providing functions for visualization '''

import matplotlib.pyplot as plt

from .util import get_projection, estimate_frame_from_ctype
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

    if frame == ('icrs', 'barycentric'):
        get_lon = lambda x: getattr(x, 'icrs').spherical.lon
        get_lat = lambda x: getattr(x, 'icrs').spherical.lat
        xlabel = 'Right Ascension'
        ylabel = 'Declination'
    elif frame in ('galactic'):
        get_lon = lambda x: getattr(x, 'gcgrs').spherical.lon
        get_lat = lambda x: getattr(x, 'gcgrs').spherical.lat
        xlabel = 'Galactic Longitude'
        ylabel = 'Galactic Latitude'
    else:
        get_lon = lambda x: getattr(x, 'gcrs').spherical.lon
        get_lat = lambda x: getattr(x, 'gcrs').spherical.lat
        xlabel = 'Right Ascension'
        ylabel = 'Declination'

    title = options.pop('title', None)
    marker = options.pop('marker', 'x')
    axis.set_aspect(1.0)
    axis.set_position([0.13, 0.10, 0.85, 0.85])
    axis.scatter(get_lon(sources),
                 get_lat(sources),
                 transform=axis.get_transform(frame),
                 marker=marker,
                 label=title,
                 **options)
    axis.grid()
    if title is not None:
        axis.legend(bbox_to_anchor=[1, 1], loc='lower right', frameon=False)
    axis.set_xlabel(xlabel, fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14)
