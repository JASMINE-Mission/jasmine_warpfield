#!/usr/bin/env python
# coding: utf-8
from astropy.coordinates import SkyCoord
from astropy.table import QTable, unique, join
from astropy.time import Time
import astropy.units as u
import jax
import numpy as np

from warpfield.telescope.source import SourceTable
from warpfield.telescope.source import convert_skycoord_to_sourcetable

jax.config.update('jax_enable_x64', True)


def propagate(reference, obstime):
    ''' Convert the reference positions at the observation epoch

    Parameters:
      reference (QTable):
        A stellar source catalog with kinematics
      obstime (Time):
        The observation epoch

    Returns:
      The reference catalog at the observation epoch. The coordinates
      are given in the Geocentric celestial coordinates, but the abberation
      offsets are removed.
    '''
    dT = obstime - Time(reference['ref_epoch'], format='jyear', scale='tcb')
    distant = reference.copy()
    distant['parallax'] = 1e-8 * u.mas
    s = SourceTable(reference).skycoord.apply_space_motion(obstime)
    t = SourceTable(distant).skycoord.apply_space_motion(obstime)
    ra = s.icrs.ra + (s.gcrs.ra - t.gcrs.ra)
    dec = s.icrs.dec + (s.gcrs.dec - t.gcrs.dec)
    skycoord = SkyCoord(ra, dec, frame='icrs', obstime=obstime)
    propagated = convert_skycoord_to_sourcetable(skycoord).table
    propagated['ra_error'] = \
        reference['ra_error'] + dT * reference['pmra_error']
    propagated['dec_error'] = \
        reference['dec_error'] + dT * reference['pmdec_error']
    propagated['source_id'] = reference['source_id']
    propagated['reference_flag'] = reference['reference_flag']
    propagated['count'] = reference['count']
    return propagated


def update_reference(reference, observation):
    unique_id = unique(observation, 'source_id')['source_id', ]
    unique_id.meta = {}
    reference = join(unique_id, reference, keys='source_id')

    count = observation['source_id', ].copy()
    count_by_id = count.group_by('source_id')
    count_by_id = QTable([
        count_by_id['source_id'].groups.aggregate(np.min),
        count_by_id['source_id'].groups.aggregate(np.size),
    ], names=[
        'source_id',
        'count',
    ])
    return join(reference, count_by_id, keys='source_id')
