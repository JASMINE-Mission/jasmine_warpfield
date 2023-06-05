#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Handling astronomical sources '''

from dataclasses import dataclass, field
from astropy.coordinates import SkyCoord, Angle, Distance
from astropy.table import QTable
from astropy.time import Time
from astroquery.gaia import Gaia
import astropy.units as u
import numpy as np

from .container import QTableContainer


__debug_mode__ = False

__columns__ = {
    'source_id': None,
    'ra': 'degree',
    'ra_error': 'mas',
    'dec': 'degree',
    'dec_error': 'mas',
    'phot_g_mean_mag': 'mag',
    'phot_bp_mean_mag': 'mag',
    'phot_rp_mean_mag': 'mag',
    'pmra': 'mas/year',
    'pmra_error': 'mas/year',
    'pmdec': 'mas/year',
    'pmdec_error': 'mas/year',
    'parallax': 'mas',
    'parallax_error': 'mas',
    'ruwe': None,
    'non_single_star': None,
    'ref_epoch': 'year',
}


def convert_skycoord_to_sourcetable(skycoord):
    source_id = np.arange(len(skycoord))
    return SourceTable(QTable([
        source_id,
        skycoord.icrs.ra,
        skycoord.icrs.dec,
    ], names=['source_id', 'ra', 'dec']))


@dataclass(frozen=True)
class SourceTable(QTableContainer):
    ''' Source Table

    Attributes:
      table (QTable):
          Table of celestial objects.
      skycoord (SkyCoord):
          Auto-generated SkyCoord object.

     The table should contain the following columns.

        - source_id: unique source ID
        - ra: right ascension in the ICRS frame
        - dec: declination in the ICRS frame

     The following columns are recognized when defining the `skycoord`.
     If they are not defined, parallax is set `None`, proper motions are
     set to zeros, and ref_epoch is set J2000.0 (TCB).

        - parallax: parallax
        - pmra: net proper motion in right ascension (μα*)
        - pmdec: net proper motion in declination (μδ)
        - ref_epoch: measurement epoch
    '''
    skycoord: SkyCoord = field(init=False)

    @staticmethod
    def __convert_epoch(time):
        return Time(time, format='decimalyear', scale='tcb')

    def __ra(self):
        ''' right ascension '''
        return self.table['ra']

    def __dec(self):
        ''' declination '''
        return self.table['dec']

    def __pmra(self):
        ''' net proper motion in right ascension

        Note:
          Returns zero mas/year if `pmra` is not defined.
        '''
        try:
            pmra = self.table['pmra']
        except KeyError:
            # proper motion is set zero if not given.
            pmra = np.zeros(len(self.table)) * u.mas / u.year
        return pmra

    def __pmdec(self):
        ''' net proper motion in declination

        Note:
          Returns zero mas/year if `pmdec` is not defined.
        '''
        try:
            pmdec = self.table['pmdec']
        except KeyError:
            # proper motion is set zero if not given.
            pmdec = np.zeros(len(self.table)) * u.mas / u.year
        return pmdec

    def __distance(self):
        ''' generate distance from parallax

        Note:
          Returns `None` if `parallax` nor `distance` is not defined.
          The `distance` column should be given as length.
        '''
        if self.has('parallax'):
            parallax = np.clip(self.table['parallax'], 1e-6 *u.mas, np.inf)
            return Distance(parallax=parallax)
        elif self.has('distance'):
            assert self.get_dimension('distance') == 'length'
            return Distance(value=self.table['distance'])
        else:
            return None

    def __epoch(self):
        ''' epoch of catalog

        Note:
          Returns J2000.0 (TCB) if `ref_epoch` nor `epoch` is not defined.
        '''
        if self.has('ref_epoch'):
            return self.__convert_epoch(self.table['ref_epoch'].data)
        elif self.has('epoch'):
            return self.__convert_epoch(self.table['epoch'].data)
        else:
            # obstime is assumed to be J2000.0 if epoch is not given.
            return self.__convert_epoch(2000.0)

    def __post_init__(self):
        assert self.has('source_id', 'ra', 'dec')
        skycoord = SkyCoord(
            ra=self.__ra(),
            dec=self.__dec(),
            pm_ra_cosdec=self.__pmra(),
            pm_dec=self.__pmdec(),
            distance=self.__distance(),
            obstime=self.__epoch())
        self.__set_skycoord(skycoord)

    def __set_skycoord(self, skycoord):
        object.__setattr__(self, 'skycoord', skycoord)


@dataclass(frozen=True)
class FocalPlanePositionTable(QTableContainer):
    ''' FocalPlanePositionTable

    Attributes:
      table (QTable):
          Table of celestial objects.

     The table should contain the following columns.

        - source_id: unique source ID
        - x: x-coordinate on the focal plane as length
        - y: y-coordinate on the focal plane as length
    '''
    def __post_init__(self):
        assert self.has('source_id', 'x', 'y')
        assert self.get_dimension('source_id') == 'dimensionless'
        assert self.get_dimension('x') == 'length'
        assert self.get_dimension('y') == 'length'


@dataclass(frozen=True)
class DetectorPositionTable(QTableContainer):
    ''' DetectorPositionTable

    Attributes:
      table (QTable):
          Table of celestial objects.

     The table should contain the following columns.

        - source_id: right ascension
        - dec: declination
        - parallax: parallax
        - pmra: proper motion in right ascension (μα*)
        - pmdec: proper motion in declination (μδ)
        - ref_epoch: measurement epoch
    '''
    def __post_init__(self):
        assert self.has('source_id', 'nx', 'ny')
        assert self.get_dimension('source_id') == 'dimensionless'
        assert self.get_dimension('nx') == 'dimensionless'
        assert self.get_dimension('ny') == 'dimensionless'


def gaia_query_builder(
        pointing, radius, snr_limit, catalog='gaiadr3.gaia_source'):
    ''' Construct a query string

    Arguments:
      pointing: A center of the search circle.
      radius: A serach radius.
      snr_limit: A lower limit of `parallax_over_error`.
      catalog: The name of catalog (default: `gaiadr3.gaia_source`)

    Returns:
      A SQL query string.
    '''
    return f'''
    SELECT
        {','.join(__columns__.keys())}
    FROM
        {catalog}
    WHERE
        1=CONTAINS(
          POINT('ICRS', {pointing.icrs.ra.deg}, {pointing.icrs.dec.deg}),
          CIRCLE('ICRS', ra, dec, {radius.deg}))
    AND
        parallax_over_error > {snr_limit}
    '''


def retrieve_gaia_sources(pointing, radius, snr_limit=10.0, row_limit=-1):
    ''' Retrive sources around (lon, lat) from Gaia EDR3 catalog

    Arguments:
      pointing (SkyCoord):
          Celestial coordinates of the search center.
      radius (float or Angle):
          A search radius in degree.
      snr_limit (float, optional):
          A lower limit of `parallax_over_error`.
      row_limit (int, optional):
          The maximum number of records.
          `-1` means no limit in the number of records.

    Return:
      A table containig souces wihtin the search circle.
    '''

    # Get an acceess to the Gaia TAP+.
    #   - Set the target table to Gaia DR3.
    #   - Remove the limit of the query number.
    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
    Gaia.ROW_LIMIT = row_limit

    if not isinstance(radius, Angle):
        radius = Angle(radius, unit=u.degree)

    pointing = pointing.transform_to('icrs')
    query = gaia_query_builder(pointing, radius, snr_limit)

    res = Gaia.launch_job_async(query)

    if __debug_mode__ is True:
        print(res)

    record = res.get_results()
    record['non_single_star'] = record['non_single_star'] > 0
    return SourceTable(QTable(record))
