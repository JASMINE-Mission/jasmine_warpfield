#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser as ap
from astropy.table import QTable
from astropy.coordinates import SkyCoord, Longitude, Latitude, Angle
import numpy as np
import pandas as pd
import astropy.units as u

from warpfield.telescope.distortion.legendre import LegendreDistortion
from warpfield.telescope.jasmine import get_jasmine
from warpfield.telescope.telescope import Detector

description = '''
JASMINE astrometry challenge example
'''.strip()

seed = 42
np.random.seed(seed)


class PositionAngle(Longitude):
    pass


def get_fields(stride):
    l0 = Longitude(-0.4 * u.deg)
    b0 = Latitude(0.2 * u.deg)
    lon = [l0, l0, l0 + 3 * stride, l0 + 4 * stride]
    lat = [b0, b0 - stride, b0 - stride, b0 - stride]
    return SkyCoord(lon, lat, frame='galactic')


def generate(fields, catalog, plate, filename):
    table = QTable.read(catalog, format='ascii.ipac')
    sources = SkyCoord(table['ra'], table['dec'], frame='icrs')

    n_coeff = 5
    coeff_x = np.zeros((n_coeff + 1, n_coeff + 1))
    coeff_y = np.zeros((n_coeff + 1, n_coeff + 1))
    coeff_x[1, 0] = +2.2
    coeff_y[0, 1] = +2.7
    coeff_x[1, 1] = +0.2
    coeff_y[1, 1] = +0.3
    coeff_x[3, 0] = +1.0
    coeff_y[0, 3] = -1.0
    coeff_x[5, 0] = +0.3
    coeff_y[0, 5] = +0.5

    n_field = []
    n_block = []
    n_plate = []
    catalog = []
    tel_ra  = []
    tel_dec = []
    tel_pa  = []
    tel_foc = []

    ## replace with a large single detector
    detector = Detector(40960, 40960, 1 * u.um)

    for n,field in enumerate(fields):
        ## calculate the position angle offset.
        pos = field.directional_offset_by(0.0 * u.deg, 0.1 * u.deg)
        pa0 = field.icrs.position_angle(pos)

        for m in range(plate):
            dl = np.random.normal(0, 5) * u.arcsec
            db = np.random.normal(0, 5) * u.arcsec
            lon = Longitude(field.galactic.l + dl)
            lat = Latitude(field.galactic.b + db)
            center = SkyCoord(lon, lat, frame='galactic')
            pa = Angle(np.random.normal(0, 1) * u.arcmin)

            ## calculate the position angle offset
            ## due to the cooridinate conversion.
            pos = center.directional_offset_by(0.0 * u.deg, 0.1 * u.deg)
            dpa = center.icrs.position_angle(pos)

            jasmine = get_jasmine(center, pa)
            r_efl = 1 + 1e-3 * np.random.normal()
            jasmine.optics.focal_length *= r_efl
            jasmine.detectors = [detector,]

            vanilla = jasmine.optics.imaging(sources)
            vanilla['catalog_id'] = vanilla.index.to_series()
            vanilla = vanilla.loc[:, ['x', 'y', 'catalog_id']]

            distortion = LegendreDistortion(n_coeff, coeff_x, coeff_y)
            jasmine.set_distortion(distortion)
            distorted = jasmine.observe(sources)[0]
            distorted['catalog_id'] = distorted.index.to_series()
            distorted['field_id'] = n
            distorted['block_id'] = m
            distorted['plate_id'] = m + n * plate
            distorted = distorted.reset_index(drop=True)
            position = distorted.merge(
                vanilla,
                on='catalog_id',
                suffixes=('', '_orig'))

            dx = position.x - position.x_orig
            dy = position.y - position.y_orig
            print(f'''ditortion plate #{n}.{m}
                max   : {dx.max()}, {dy.max()}
                min   : {dx.min()}, {dy.min()}
                mean  : {dx.mean()}, {dy.mean()}
                median: {dx.median()}, {dy.median()}
                stddev: {dx.std()}, {dy.std()}''')

            catalog.append(position)
            n_field.append(n)
            n_block.append(m)
            n_plate.append(m + n * plate)
            tel_ra.append(center.icrs.ra.deg)
            tel_dec.append(center.icrs.dec.deg)
            tel_pa.append(PositionAngle(pa + dpa).deg)
            tel_foc.append(jasmine.optics.focal_length.to_value('m'))

    ## compile catalog
    catalog = pd.concat(catalog)

    keywords = {
        'pointing_ra': {
            'value': fields.icrs.ra.deg
        },
        'pointing_dec': {
            'value': fields.icrs.dec.deg
        },
        'position_angle': {
            'value': pa0.deg
        },
    }

    table = QTable(
        [
            catalog.x.array * u.um,
            catalog.y.array * u.um,
            catalog.catalog_id.array,
            catalog.field_id.array,
            catalog.block_id.array,
            catalog.plate_id.array,
            catalog.x_orig.array * u.um,
            catalog.y_orig.array * u.um,
            catalog.ra.array * u.deg,
            catalog.dec.array * u.deg,
        ],
        names=(
            'x', 'y', 'catalog_id', 'field_id', 'block_id',
            'plate_id', 'x_orig', 'y_orig', 'ra', 'dec'),
        meta={
            'keywords': keywords,
            'comments': [
                description,
              ],
        })
    print(table)
    table.write(filename, format='ascii.ipac', overwrite=True)

    table = QTable([
        n_field,
        n_block,
        n_plate,
        np.array(tel_ra) * u.deg,
        np.array(tel_dec) * u.deg,
        np.array(tel_pa) * u.deg,
        np.array(tel_foc) * u.m,
    ],
        names=('field', 'block', 'plate', 'ra', 'dec', 'pa', 'foc'),
        meta={'comments': ['The attitudes of the telescope.']},
    )
    print(table)
    filename = filename.replace('.txt', '_pointing.txt')
    table.write(filename, format='ascii.ipac', overwrite=True)

    with open('sip.npz', 'wb') as f:
        np.savez(f, coeff_x=coeff_x, coeff_y=coeff_y)


if __name__ == '__main__':
    parser = ap(description='Generate baseline challenges')

    parser.add_argument(
        'output', type=str, help='output filename')
    parser.add_argument(
        '-n', '--n_plate', type=int, default=4,
        help='the number of plates per field')
    parser.add_argument(
        '--catalog', type=str, default='source_list.txt',
        help='filename of the source list')

    args = parser.parse_args()


    stride = Angle(15 * u.arcmin)
    fields = get_fields(stride)

    plate = args.n_plate
    catalog = args.catalog
    filename = args.output
    generate(fields, catalog, plate, filename)
