#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Apparent source catalog for astrometric analysis '''

from astropy.table import QTable
import astropy.units as u
from jax import Array
import jax.numpy as jnp
import numpy as np
import zodiax as zdx


__all__ = ['SourceCatalog']


def _optional_array(value, name, shape, *, non_negative=False):
    ''' Validate one optional array-valued catalog attribute '''
    if value is None:
        return None
    value = jnp.asarray(value, dtype=float)
    if value.shape != shape:
        raise ValueError(f'`{name}` should have the same shape as `ra`.')
    numeric = np.asarray(value)
    if np.any(~np.isfinite(numeric)):
        raise ValueError(f'`{name}` should contain finite values.')
    if non_negative and np.any(numeric < 0):
        raise ValueError(f'`{name}` should be non-negative.')
    return value


def _select_optional(value, index):
    ''' Select an optional catalog attribute '''
    return None if value is None else value[index]


class SourceCatalog(zdx.Base):
    ''' Celestial source positions represented as a PyTree

    Attributes:
        ra: Right ascensions in degrees with shape ``(N_source,)``.
        dec: Declinations in degrees with shape ``(N_source,)``.
        magnitude: Optional magnitudes with shape ``(N_source,)``.
        magnitude_error: Optional magnitude uncertainties in magnitudes.
        ra_error: Optional right-ascension uncertainties in degrees.
        dec_error: Optional declination uncertainties in degrees.
    '''

    ra: Array
    dec: Array
    magnitude: Array | None
    magnitude_error: Array | None
    ra_error: Array | None
    dec_error: Array | None

    def __init__(
            self, ra, dec, *, magnitude=None, magnitude_error=None,
            ra_error=None, dec_error=None):
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.ndim != 1:
            raise ValueError('`dec` should be a one-dimensional array.')
        if ra.shape != dec.shape:
            raise ValueError('`ra` and `dec` should have the same shape.')

        magnitude = _optional_array(magnitude, 'magnitude', ra.shape)
        magnitude_error = _optional_array(
            magnitude_error,
            'magnitude_error',
            ra.shape,
            non_negative=True,
        )
        ra_error = _optional_array(
            ra_error, 'ra_error', ra.shape, non_negative=True)
        dec_error = _optional_array(
            dec_error, 'dec_error', ra.shape, non_negative=True)

        self.ra = ra
        self.dec = dec
        self.magnitude = magnitude
        self.magnitude_error = magnitude_error
        self.ra_error = ra_error
        self.dec_error = dec_error

    def __len__(self):
        return self.ra.shape[0]

    def __getitem__(self, index):
        ''' Select sources while preserving the catalog dimension '''
        index = np.atleast_1d(np.arange(len(self))[index])
        return SourceCatalog(
            self.ra[index],
            self.dec[index],
            magnitude=_select_optional(self.magnitude, index),
            magnitude_error=_select_optional(self.magnitude_error, index),
            ra_error=_select_optional(self.ra_error, index),
            dec_error=_select_optional(self.dec_error, index),
        )

    def __iter__(self):
        ''' Iterate over single-source catalogs '''
        for index in range(len(self)):
            yield self[index]

    def to_qtable(self):
        ''' Convert catalog attributes into a unit-aware QTable '''
        table = QTable({
            'source_id': np.arange(len(self), dtype=int),
            'ra': np.asarray(self.ra) * u.deg,
            'dec': np.asarray(self.dec) * u.deg,
        })
        units = {
            'magnitude': u.mag,
            'magnitude_error': u.mag,
            'ra_error': u.deg,
            'dec_error': u.deg,
        }
        for name, unit in units.items():
            value = getattr(self, name)
            if value is not None:
                table[name] = np.asarray(value) * unit
        return table

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

        optional = {}
        units = {
            'magnitude': u.mag,
            'magnitude_error': u.mag,
            'ra_error': u.deg,
            'dec_error': u.deg,
        }
        try:
            for name, unit in units.items():
                optional[name] = (
                    u.Quantity(table[name]).to_value(unit)
                    if name in table.colnames else None
                )
        except u.UnitConversionError as error:
            raise ValueError(
                'Optional source attributes have incompatible units.'
            ) from error
        return cls(ra, dec, **optional)

    def take(self, index):
        ''' Select source positions using an integer index array '''
        return self.ra[index], self.dec[index]
