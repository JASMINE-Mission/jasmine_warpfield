#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Source catalog parameters for astrometric analysis '''

from dataclasses import dataclass

from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity
import astropy.units as u
from jax import Array
import jax.numpy as jnp
import numpy as np
import zodiax as zdx

from .observer import Observer


__all__ = ['AstrometricCatalog', 'SourceCatalog']


@dataclass(frozen=True, init=False)
class AstrometricCatalog:
    ''' Astrometric source parameters at a reference epoch

    Attributes:
        ra: Right ascensions with shape ``(N_source,)``.
        dec: Declinations with shape ``(N_source,)``.
        pm_ra_cosdec: Proper motions in right ascension including the cosine
            declination factor, with shape ``(N_source,)``.
        pm_dec: Proper motions in declination with shape ``(N_source,)``.
        parallax: Annual parallaxes with shape ``(N_source,)``.
        epoch: Reference epoch of the catalog.
    '''

    ra: Angle
    dec: Angle
    pm_ra_cosdec: Quantity
    pm_dec: Quantity
    parallax: Quantity
    epoch: Time

    def __init__(
            self, ra, dec, pm_ra_cosdec, pm_dec, parallax, epoch):
        ra = Angle(ra, unit=u.deg)
        dec = Angle(dec, unit=u.deg)
        pm_ra_cosdec = Quantity(pm_ra_cosdec, unit=u.mas / u.yr)
        pm_dec = Quantity(pm_dec, unit=u.mas / u.yr)
        parallax = Quantity(parallax, unit=u.mas)
        epoch = Time(epoch)

        shape = ra.shape
        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        for name, value in (
                ('dec', dec),
                ('pm_ra_cosdec', pm_ra_cosdec),
                ('pm_dec', pm_dec),
                ('parallax', parallax)):
            if value.shape != shape:
                raise ValueError(
                    f'`{name}` should have the same shape as `ra`.')
        if not epoch.isscalar and epoch.shape != shape:
            raise ValueError(
                '`epoch` should be scalar or have the same shape as `ra`.')
        if np.any(~np.isfinite(parallax.to_value(u.mas))):
            raise ValueError('`parallax` should contain finite values.')
        if np.any(parallax < 0 * u.mas):
            raise ValueError('`parallax` should be non-negative.')

        # Constructing SkyCoord here also validates the declination range and
        # the compatibility of the supplied Astropy quantities.
        self._to_skycoord(ra, dec, pm_ra_cosdec, pm_dec, parallax, epoch)

        object.__setattr__(self, 'ra', ra)
        object.__setattr__(self, 'dec', dec)
        object.__setattr__(self, 'pm_ra_cosdec', pm_ra_cosdec)
        object.__setattr__(self, 'pm_dec', pm_dec)
        object.__setattr__(self, 'parallax', parallax)
        object.__setattr__(self, 'epoch', epoch)

    @staticmethod
    def _to_skycoord(
            ra, dec, pm_ra_cosdec, pm_dec, parallax, epoch):
        return SkyCoord(
            ra=ra,
            dec=dec,
            pm_ra_cosdec=pm_ra_cosdec,
            pm_dec=pm_dec,
            distance=Distance(parallax=parallax),
            obstime=epoch,
            frame='icrs',
        )

    @property
    def skycoord(self):
        ''' Return the catalog as an ICRS SkyCoord '''
        return self._to_skycoord(
            self.ra,
            self.dec,
            self.pm_ra_cosdec,
            self.pm_dec,
            self.parallax,
            self.epoch,
        )

    def __len__(self):
        return self.ra.shape[0]

    def __getitem__(self, index):
        ''' Select sources while preserving the catalog dimension '''
        index = np.atleast_1d(np.arange(len(self))[index])
        epoch = self.epoch if self.epoch.isscalar else self.epoch[index]
        return AstrometricCatalog(
            self.ra[index],
            self.dec[index],
            self.pm_ra_cosdec[index],
            self.pm_dec[index],
            self.parallax[index],
            epoch,
        )

    def __iter__(self):
        ''' Iterate over single-source catalogs '''
        for index in range(len(self)):
            yield self[index]

    def to_qtable(self):
        ''' Convert catalog attributes into a unit-aware QTable '''
        epoch = self.epoch
        if epoch.isscalar:
            epoch = epoch + np.zeros(len(self)) * u.day
        return QTable({
            'source_id': np.arange(len(self), dtype=int),
            'ra': self.ra,
            'dec': self.dec,
            'pm_ra_cosdec': self.pm_ra_cosdec,
            'pm_dec': self.pm_dec,
            'parallax': self.parallax,
            'epoch': epoch,
        })

    @classmethod
    def from_qtable(cls, table):
        ''' Construct a catalog from a unit-aware QTable '''
        cls._validate_qtable(table)
        return cls(
            ra=table['ra'],
            dec=table['dec'],
            pm_ra_cosdec=table['pm_ra_cosdec'],
            pm_dec=table['pm_dec'],
            parallax=table['parallax'],
            epoch=Time(table['epoch']),
        )

    @staticmethod
    def _validate_qtable(table):
        if not isinstance(table, QTable):
            raise TypeError('`table` should be a QTable instance.')
        required = (
            'ra',
            'dec',
            'pm_ra_cosdec',
            'pm_dec',
            'parallax',
            'epoch',
        )
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise ValueError(
                '`table` is missing required columns: '
                + ', '.join(missing))

    def propagate(self, observer):
        ''' Generate apparent source positions in an observer frame '''
        if not isinstance(observer, Observer):
            raise TypeError('`observer` should be an Observer instance.')
        if not hasattr(observer, 'obstime'):
            raise TypeError('`observer` should define `obstime`.')

        coordinate = self.skycoord.apply_space_motion(
            new_obstime=observer.obstime,
        )
        apparent = coordinate.transform_to(observer)
        return SourceCatalog(
            apparent.spherical.lon.to_value(u.deg),
            apparent.spherical.lat.to_value(u.deg),
        )


class SourceCatalog(zdx.Base):
    ''' Celestial source positions represented as a PyTree

    Attributes:
        ra: Right ascensions in degrees with shape ``(N_source,)``.
        dec: Declinations in degrees with shape ``(N_source,)``.
    '''

    ra: Array
    dec: Array

    def __init__(self, ra, dec):
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.ndim != 1:
            raise ValueError('`dec` should be a one-dimensional array.')
        if ra.shape != dec.shape:
            raise ValueError('`ra` and `dec` should have the same shape.')

        self.ra = ra
        self.dec = dec

    def __len__(self):
        return self.ra.shape[0]

    def __getitem__(self, index):
        ''' Select sources while preserving the catalog dimension '''
        index = np.atleast_1d(np.arange(len(self))[index])
        return SourceCatalog(self.ra[index], self.dec[index])

    def __iter__(self):
        ''' Iterate over single-source catalogs '''
        for index in range(len(self)):
            yield self[index]

    def to_qtable(self):
        ''' Convert catalog attributes into a unit-aware QTable '''
        return QTable({
            'source_id': np.arange(len(self), dtype=int),
            'ra': np.asarray(self.ra) * u.deg,
            'dec': np.asarray(self.dec) * u.deg,
        })

    @classmethod
    def from_qtable(cls, table):
        ''' Construct a catalog from a unit-aware QTable '''
        if not isinstance(table, QTable):
            raise TypeError('`table` should be a QTable instance.')
        missing = [
            name for name in ('ra', 'dec') if name not in table.colnames]
        if missing:
            raise ValueError(
                '`table` is missing required columns: '
                + ', '.join(missing))
        try:
            ra = u.Quantity(table['ra']).to_value(u.deg)
            dec = u.Quantity(table['dec']).to_value(u.deg)
        except u.UnitConversionError as error:
            raise ValueError(
                '`ra` and `dec` should have angular units.') from error
        return cls(ra, dec)

    def take(self, index):
        ''' Select source positions using an integer index array '''
        return self.ra[index], self.dec[index]
