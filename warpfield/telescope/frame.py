#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Providing celestial frames '''

from astropy.coordinates.builtin_frames import utils
from astropy.coordinates.builtin_frames import icrs_fk5_transforms
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates import Galactic, GCRS
from astropy.coordinates import CartesianRepresentationAttribute as CRA
from astropy.coordinates import TimeAttribute
from astropy.coordinates import DynamicMatrixTransform
from astropy.coordinates import frame_transform_graph

# import astropy.coordinates.representation as r
import astropy.units as u


class GCGRS(Galactic):
    ''' A Geocentric coordinate or frame in the Galactic coordinate system

    This frame is defined based on the `Galactic` frame. The observer is
    located to the center of the Earth. The relative velocity to the Earth
    is set zero.
    '''

    obstime = TimeAttribute(default=utils.DEFAULT_OBSTIME)
    obsgeoloc = CRA(default=[0, 0, 0], unit=u.m)
    obsgeovel = CRA(default=[0, 0, 0], unit=u.m / u.s)


@frame_transform_graph.transform(DynamicMatrixTransform, GCRS, GCGRS)
def gcrs_to_gcgrs(gcrscoord, gcgrsframe):
    # First, rotate the frame from ICRS to FK5 without correcting precession.
    # Then, the frame is aligned to the Galactic coordiante.
    return (
        rotation_matrix(180 - GCGRS._lon0_J2000.degree, "z")
        @ rotation_matrix(90 - GCGRS._ngp_J2000.dec.degree, "y")
        @ rotation_matrix(GCGRS._ngp_J2000.ra.degree, "z")
        @ icrs_fk5_transforms._ICRS_TO_FK5_J2000_MAT
    )


@frame_transform_graph.transform(DynamicMatrixTransform, GCGRS, GCRS)
def gcgrs_to_gcrs(gcgrscoord, gcrsframe):
    return matrix_transpose(gcrs_to_gcgrs(gcrsframe, gcgrscoord))
