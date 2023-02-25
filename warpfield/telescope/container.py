#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Define QTableContainer '''

from dataclasses import dataclass
from astropy.table import QTable
import astropy.io.fits as fits
import astropy.units as u


@dataclass(frozen=True)
class QTableContainer:
    ''' QTable with I/O functions

    Attributes:
      table (QTable):
          Table of celestial objects.
    '''
    table: QTable

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return self.table[key]

    def has(self, *items):
        names = self.table.colnames
        return all([name in names for name in items])

    def get_dimension(self, name):
        return u.get_physical_type(self.table[name])

    @classmethod
    def from_fitsfile(cls, filename, key='table'):
        ''' Generate a SourceTable from a FITS file

        Arguments:
          filename (str):
              The path to the source FITS file.
          key (str):
              The name of the FITS extension containing the table data.

        Returns:
          A table instance.
        '''
        hdul = fits.open(filename)
        table = QTable.read(hdul[key])
        return cls(table=table)

    def writeto(self, filename, overwrite=False):
        ''' Dump a SourceTable into a FITS file

        Arguments:
          filename (str):
              The path to the output filename.

        Options:
          overwrite (bool):
              An existing file will be overwritten if true.
        '''
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(data=self.table, name='table')
        ])
        hdul.writeto(filename, overwrite=overwrite)
