#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Telescope pointing parameters for astrometric analysis"""

from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable
import astropy.units as u
from jax import Array
import jax.numpy as jnp
import numpy as np
import zodiax as zdx


__all__ = ['Pointing']


class Pointing(zdx.Base):
    """Telescope pointings represented as a PyTree

    Attributes:
        ra: Right ascensions in degrees with shape ``(N_exposure,)``.
        dec: Declinations in degrees with shape ``(N_exposure,)``.
        position_angle: Position angles in degrees with shape
            ``(N_exposure,)``.
    """

    ra: Array
    dec: Array
    position_angle: Array

    def __init__(self, ra, dec, position_angle):
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)
        position_angle = jnp.asarray(position_angle, dtype=float)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.shape != ra.shape:
            raise ValueError('`dec` should have the same shape as `ra`.')
        if position_angle.shape != ra.shape:
            raise ValueError(
                '`position_angle` should have the same shape as `ra`.'
            )

        self.ra = ra
        self.dec = dec
        self.position_angle = position_angle

    def __len__(self):
        return self.ra.shape[0]

    def __getitem__(self, index):
        """Select pointings while preserving the collection dimension"""
        return Pointing(
            jnp.atleast_1d(self.ra[index]),
            jnp.atleast_1d(self.dec[index]),
            jnp.atleast_1d(self.position_angle[index]),
        )

    def __iter__(self):
        """Iterate over single-pointing collections"""
        for index in range(len(self)):
            yield self[index]

    @classmethod
    def from_coord(cls, frame, lon, lat, pa):
        """Convert an ICRS or Galactic attitude into an ICRS pointing

        ``lon``, ``lat``, and ``pa`` are interpreted as degrees when given
        without units. The position angle is measured east of north in the
        specified frame.
        """
        if frame not in ('icrs', 'galactic'):
            raise ValueError('`frame` should be either "icrs" or "galactic".')

        angles = [
            np.asarray(Angle(value, unit=u.deg).to_value(u.deg))
            for value in (lon, lat, pa)
        ]
        if any(value.ndim > 1 for value in angles):
            raise ValueError(
                '`lon`, `lat`, and `pa` should be scalar or one-dimensional.'
            )
        try:
            lon, lat, pa = np.broadcast_arrays(*angles)
        except ValueError as error:
            raise ValueError(
                '`lon`, `lat`, and `pa` should have compatible shapes.'
            ) from error
        if lon.ndim == 0:
            lon, lat, pa = (value.reshape((1,)) for value in (lon, lat, pa))

        coordinate = SkyCoord(lon, lat, unit=u.deg, frame=frame)
        direction = coordinate.directional_offset_by(
            Angle(pa, unit=u.deg),
            1 * u.arcsec,
        )
        icrs = coordinate.icrs
        direction = direction.icrs
        icrs_position_angle = icrs.position_angle(direction)
        return cls(
            icrs.ra.to_value(u.deg),
            icrs.dec.to_value(u.deg),
            icrs_position_angle.to_value(u.deg),
        )

    def to_qtable(self):
        """Convert pointing parameters into a unit-aware QTable"""
        return QTable({
            'exposure_id': np.arange(len(self), dtype=int),
            'ra': np.asarray(self.ra) * u.deg,
            'dec': np.asarray(self.dec) * u.deg,
            'position_angle': np.asarray(self.position_angle) * u.deg,
        })

    @classmethod
    def from_qtable(cls, table):
        """Construct pointing parameters from a unit-aware QTable"""
        if not isinstance(table, QTable):
            raise TypeError('`table` should be a QTable instance.')
        required = ('ra', 'dec', 'position_angle')
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise ValueError(
                '`table` is missing required columns: ' + ', '.join(missing)
            )
        try:
            values = [
                u.Quantity(table[name]).to_value(u.deg) for name in required
            ]
        except u.UnitConversionError as error:
            raise ValueError(
                'Pointing coordinates should have angular units.'
            ) from error
        return cls(*values)

    def take(self, index):
        """Select pointing parameters using an integer index array"""
        return (
            self.ra[index],
            self.dec[index],
            self.position_angle[index],
        )
