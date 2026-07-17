#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Reference-epoch astrometric source catalog '''

from dataclasses import dataclass

from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity
import astropy.units as u
import numpy as np

from ..observer import Observer
from .source import SourceCatalog


__all__ = ['AstrometricCatalog']


def _optional_quantity(value, unit, name, shape, *, non_negative=False):
    ''' Validate one optional unit-aware catalog attribute '''
    if value is None:
        return None
    value = Quantity(value, unit=unit)
    if value.shape != shape:
        raise ValueError(f'`{name}` should have the same shape as `ra`.')
    numeric = value.to_value(unit)
    if np.any(~np.isfinite(numeric)):
        raise ValueError(f'`{name}` should contain finite values.')
    if non_negative and np.any(numeric < 0):
        raise ValueError(f'`{name}` should be non-negative.')
    return value


def _select_optional(value, index):
    ''' Select an optional catalog attribute '''
    return None if value is None else value[index]


@dataclass(frozen=True, init=False)
class AstrometricCatalog:
    ''' Astrometric source parameters at a reference epoch

    The catalog stores one-dimensional uncertainties but not covariances.
    During propagation, the errors in right ascension, declination, proper
    motion, and parallax are therefore treated as independent. In geometric
    terms, the input positional uncertainty is approximated by an ellipse
    whose axes are parallel to the right-ascension and declination axes.

    Attributes:
        ra: Right ascensions with shape ``(N_source,)``.
        dec: Declinations with shape ``(N_source,)``.
        pm_ra_cosdec: Proper motions in right ascension including the cosine
            declination factor, with shape ``(N_source,)``.
        pm_dec: Proper motions in declination with shape ``(N_source,)``.
        parallax: Annual parallaxes with shape ``(N_source,)``.
        epoch: Reference epoch of the catalog.
        magnitude: Optional magnitudes with shape ``(N_source,)``.
        magnitude_error: Optional magnitude uncertainties.
        ra_error: Optional right-ascension uncertainties.
        dec_error: Optional declination uncertainties.
        pm_ra_cosdec_error: Optional proper-motion uncertainties in right
            ascension including the cosine declination factor.
        pm_dec_error: Optional proper-motion uncertainties in declination.
        parallax_error: Optional parallax uncertainties.
    '''

    ra: Angle
    dec: Angle
    pm_ra_cosdec: Quantity
    pm_dec: Quantity
    parallax: Quantity
    epoch: Time
    magnitude: Quantity | None
    magnitude_error: Quantity | None
    ra_error: Quantity | None
    dec_error: Quantity | None
    pm_ra_cosdec_error: Quantity | None
    pm_dec_error: Quantity | None
    parallax_error: Quantity | None

    def __init__(
            self, ra, dec, pm_ra_cosdec, pm_dec, parallax, epoch, *,
            magnitude=None, magnitude_error=None, ra_error=None,
            dec_error=None, pm_ra_cosdec_error=None, pm_dec_error=None,
            parallax_error=None):
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

        magnitude = _optional_quantity(
            magnitude, u.mag, 'magnitude', shape)
        magnitude_error = _optional_quantity(
            magnitude_error,
            u.mag,
            'magnitude_error',
            shape,
            non_negative=True,
        )
        ra_error = _optional_quantity(
            ra_error, u.mas, 'ra_error', shape, non_negative=True)
        dec_error = _optional_quantity(
            dec_error, u.mas, 'dec_error', shape, non_negative=True)
        pm_ra_cosdec_error = _optional_quantity(
            pm_ra_cosdec_error,
            u.mas / u.yr,
            'pm_ra_cosdec_error',
            shape,
            non_negative=True,
        )
        pm_dec_error = _optional_quantity(
            pm_dec_error,
            u.mas / u.yr,
            'pm_dec_error',
            shape,
            non_negative=True,
        )
        parallax_error = _optional_quantity(
            parallax_error,
            u.mas,
            'parallax_error',
            shape,
            non_negative=True,
        )

        # Constructing SkyCoord here also validates the declination range and
        # the compatibility of the supplied Astropy quantities.
        self._to_skycoord(ra, dec, pm_ra_cosdec, pm_dec, parallax, epoch)

        object.__setattr__(self, 'ra', ra)
        object.__setattr__(self, 'dec', dec)
        object.__setattr__(self, 'pm_ra_cosdec', pm_ra_cosdec)
        object.__setattr__(self, 'pm_dec', pm_dec)
        object.__setattr__(self, 'parallax', parallax)
        object.__setattr__(self, 'epoch', epoch)
        object.__setattr__(self, 'magnitude', magnitude)
        object.__setattr__(self, 'magnitude_error', magnitude_error)
        object.__setattr__(self, 'ra_error', ra_error)
        object.__setattr__(self, 'dec_error', dec_error)
        object.__setattr__(
            self, 'pm_ra_cosdec_error', pm_ra_cosdec_error)
        object.__setattr__(self, 'pm_dec_error', pm_dec_error)
        object.__setattr__(self, 'parallax_error', parallax_error)

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

    def _apparent(self, observer, **updates):
        ''' Propagate astrometric parameters into an observer frame '''
        values = {
            'ra': self.ra,
            'dec': self.dec,
            'pm_ra_cosdec': self.pm_ra_cosdec,
            'pm_dec': self.pm_dec,
            'parallax': self.parallax,
        }
        values.update(updates)
        coordinate = self._to_skycoord(
            **values,
            epoch=self.epoch,
        ).apply_space_motion(new_obstime=observer.obstime)
        return coordinate.transform_to(observer)

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
            magnitude=_select_optional(self.magnitude, index),
            magnitude_error=_select_optional(self.magnitude_error, index),
            ra_error=_select_optional(self.ra_error, index),
            dec_error=_select_optional(self.dec_error, index),
            pm_ra_cosdec_error=_select_optional(
                self.pm_ra_cosdec_error, index),
            pm_dec_error=_select_optional(self.pm_dec_error, index),
            parallax_error=_select_optional(self.parallax_error, index),
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
        table = QTable({
            'source_id': np.arange(len(self), dtype=int),
            'ra': self.ra,
            'dec': self.dec,
            'pm_ra_cosdec': self.pm_ra_cosdec,
            'pm_dec': self.pm_dec,
            'parallax': self.parallax,
            'epoch': epoch,
        })
        for name in (
                'magnitude',
                'magnitude_error',
                'ra_error',
                'dec_error',
                'pm_ra_cosdec_error',
                'pm_dec_error',
                'parallax_error'):
            value = getattr(self, name)
            if value is not None:
                table[name] = value
        return table

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
            **{
                name: table[name] if name in table.colnames else None
                for name in (
                    'magnitude',
                    'magnitude_error',
                    'ra_error',
                    'dec_error',
                    'pm_ra_cosdec_error',
                    'pm_dec_error',
                    'parallax_error',
                )
            },
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

    def _propagate_errors(self, observer, apparent):
        ''' Propagate independent one-sigma errors into apparent coordinates

        The input covariance is approximated as diagonal in ``ra``, ``dec``,
        ``pm_ra_cosdec``, ``pm_dec``, and ``parallax``. Each available
        parameter is displaced by its positive one-sigma uncertainty,
        propagated independently, and the resulting longitude and latitude
        shifts are combined in quadrature. Correlations and the rotation of
        the resulting error ellipse are not represented.
        '''
        errors = (
            ('ra', self.ra_error),
            ('dec', self.dec_error),
            ('pm_ra_cosdec', self.pm_ra_cosdec_error),
            ('pm_dec', self.pm_dec_error),
            ('parallax', self.parallax_error),
        )
        variance_lon = np.zeros(len(self))
        variance_lat = np.zeros(len(self))
        available = False
        for name, error in errors:
            if error is None:
                continue
            available = True
            perturbed = self._apparent(
                observer,
                **{name: getattr(self, name) + error},
            )
            delta_lon = Angle(
                perturbed.spherical.lon - apparent.spherical.lon
            ).wrap_at(180 * u.deg).to_value(u.deg)
            delta_lat = (
                perturbed.spherical.lat - apparent.spherical.lat
            ).to_value(u.deg)
            variance_lon += delta_lon**2
            variance_lat += delta_lat**2
        if not available:
            return None, None
        return np.sqrt(variance_lon), np.sqrt(variance_lat)

    def propagate(self, observer):
        ''' Generate apparent source positions in an observer frame

        Magnitudes and their uncertainties are copied unchanged. Position,
        proper-motion, and parallax errors are propagated numerically by
        treating their covariance as diagonal and combining their independent
        effects in quadrature. Thus, the input positional error ellipse is
        assumed to be aligned with the right-ascension and declination axes.
        The output contains only axis-aligned longitude and latitude errors;
        correlations and the orientation of the propagated ellipse are
        discarded.
        '''
        if not isinstance(observer, Observer):
            raise TypeError('`observer` should be an Observer instance.')
        if not hasattr(observer, 'obstime'):
            raise TypeError('`observer` should define `obstime`.')

        apparent = self._apparent(observer)
        ra_error, dec_error = self._propagate_errors(observer, apparent)
        return SourceCatalog(
            apparent.spherical.lon.to_value(u.deg),
            apparent.spherical.lat.to_value(u.deg),
            magnitude=None if self.magnitude is None else (
                self.magnitude.to_value(u.mag)),
            magnitude_error=None if self.magnitude_error is None else (
                self.magnitude_error.to_value(u.mag)),
            ra_error=ra_error,
            dec_error=dec_error,
        )
