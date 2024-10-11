#!/usr/bin/env python
# coding: utf-8
from numpyro.infer import Predictive, SVI, Trace_ELBO
import numpyro
import jax
import jax.random as random

from astropy.table import QTable, unique
from astropy.time import Time
import astropy.io.fits as fits
import astropy.units as u
import numpy as np

from datetime import datetime

from tools.propagate import propagate, update_reference
from tools.resume import setup_params

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

jax.config.update('jax_enable_x64', True)


if __name__ == '__main__':
    from argparse import ArgumentParser as ap
    parser = ap(
        description='JASMINE Small-Scale Survey Simulation')

    parser.add_argument(
        'obs', type=str,
        help='observation file')
    parser.add_argument(
        'ref', type=str,
        help='reference catalog file')
    parser.add_argument(
        'num_iteration', type=int,
        help='number of iterations in optimization.')
    parser.add_argument(
        'output', type=str,
        help='output resluts')
    parser.add_argument(
        '-m', '--model', type=str,
        choices=('f', 'a', 'p', 'd', 's'), default='f',
        help='optimization model')
    parser.add_argument(
        '-P', '--progress', action='store_true',
        help='show progress bar')
    parser.add_argument(
        '-r', '--resume', type=str,
        help='intermediate result file')
    parser.add_argument(
        '--step', type=float, default=1e-3,
        help='initial step size in optimization')
    parser.add_argument(
        '--epsilon', type=float, default=1e-4,
        help='epsilon parameter of Adam optimizer')
    parser.add_argument(
        '-b1', '--adam-b1', type=float, default=0.99,
        help='b1 parameter of Adam optimizer')
    parser.add_argument(
        '-b2', '--adam-b2', type=float, default=0.999,
        help='b2 parameter of Adam optimizer')
    parser.add_argument(
        '-N', '--num-sample', type=int, default=200,
        help='number of samples for prediction')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed for data preparation')
    parser.add_argument(
        '-f', '--overwrite', action='store_true',
        help='overwrite the output file if exists')
    parser.add_argument(
        '-n', '--test', action='store_true',
        help='reduce the number of sources for test')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose debug messages')

    args = parser.parse_args()

    obj = QTable.read(args.obs, hdu='measurement')
    env = QTable.read(args.obs, hdu='environment')
    ref = QTable.read(args.ref)

    if args.test:
        source_id = unique(obj, 'source_id')['source_id'][::20]
        obj = obj[np.isin(obj['source_id'], source_id)]

    ref = update_reference(ref, obj)
    ref_flag = ref['reference_flag']

    datestr  = datetime.now().strftime('%Y%m%d%H%M')
    obstime = Time(env['obstime'].mean(), format='mjd', scale='tcb')

    params = {'obstime': obstime}
    if args.resume:
        params = setup_params(params, args.resume)

    if args.model == 'f':
        from tools.model.full import generate
    elif args.model == 'a':
        from tools.model.attitude import generate
    elif args.model == 'p':
        from tools.model.position import generate
    elif args.model == 'd':
        from tools.model.distortion import generate
    elif args.model == 's':
        from tools.model.sigma import generate
    else:
        raise RuntimeError(f'unexpected model "{args.model}"')
    model, guide = generate(obj, env, ref, params=params)

    opt = numpyro.optim.Adam(
        step_size=args.step, eps=args.epsilon,
        b1=args.adam_b1, b2=args.adam_b2)
    svi = SVI(model, guide, opt, loss=Trace_ELBO())

    rng_key, gen_key = random.split(random.PRNGKey(args.seed))
    result = svi.run(rng_key, args.num_iteration, progress_bar=args.progress)

    print('final loss: {}'.format(result.losses[-1]))

    params = result.params
    losses = result.losses

    pred = Predictive(
        model, guide=guide, params=params, num_samples=args.num_sample)
    sample = pred(gen_key)

    if args.verbose:
        for k, v in params.items():
            print(f'## {k}')
            print(v)

    phead = fits.Header()
    phead['input'] = args.obs, \
        'observation file'
    phead['refcat'] = args.ref, \
        'reference catalog file'
    phead['niter'] = args.num_iteration, \
        'number of iteration'
    phead['resume'] = args.resume or 'Null', \
        'intermediate result file'
    phead['step'] = args.step, \
        'step paremter of Adam optimizer'
    phead['epsilon'] = args.epsilon, \
        'epsilon parameter of Adam optimizer'
    phead['seed'] = args.seed, \
        'seed of the primary random generator key'
    phead['rng_key'] = str(rng_key), \
        'primary random generator key of JAX'
    phead['gen_key'] = str(gen_key), \
        'sampling random generator key of JAX'

    rhead = fits.Header()
    rhead['obstime'] = obstime.to_value('mjd'), \
        'observation epoch in MJD (TCB)'
    rhead['sigma_x'] = float(params['sig_x_loc'].mean()), \
        'estimated measurement uncertainty along naxis1'
    rhead['sigma_y'] = float(params['sig_y_loc'].mean()), \
        'estimated measurement uncertainty along naxis2'
    rhead['sigma'] = (rhead['sigma_x'] + rhead['sigma_y']) / 2, \
        'estimated measurement uncertainty'
    rhead['loss'] = float(losses[-1]), \
        'final ELBO loss value'
    rhead['sigma_ra'] = float(sample['sigma_a'].mean()), \
        'estimated measurement uncertainty'
    rhead['sigma_de'] = float(sample['sigma_d'].mean()), \
        'estimated measurement uncertainty'

    ref = propagate(ref, obstime)
    result = QTable([
        ref['source_id'],
        params['src_a_loc'] * u.degree,
        params['src_a_sig'] * u.degree,
        params['src_d_loc'] * u.degree,
        params['src_d_sig'] * u.degree,
        ref['ra'],
        ref['dec'],
        ref['count'],
        ref['reference_flag'],
    ], names=[
        'source_id',
        'src_a_loc',
        'src_a_sig',
        'src_d_loc',
        'src_d_sig',
        'ra_ref',
        'dec_ref',
        'num_measurement',
        'reference_flag',
    ])
    sample = QTable([
        obj['source_id'],
        obj['detector_id'],
        obj['orbit_id'],
        obj['field_id'],
        obj['plate_id'],
        sample['nx'].mean(axis=0),
        sample['ny'].mean(axis=0),
        sample['ij'].mean(axis=0)[:, 0],
        sample['ij'].mean(axis=0)[:, 1],
        sample['xy'].mean(axis=0)[:, 0],
        sample['xy'].mean(axis=0)[:, 1],
        sample['pq'].mean(axis=0)[:, 0],
        sample['pq'].mean(axis=0)[:, 1],
        params['sig_x_loc'] * u.pixel,
        params['sig_y_loc'] * u.pixel,
    ], names=[
        'source_id',
        'detector_id',
        'orbit_id',
        'field_id',
        'plate_id',
        'nx',
        'ny',
        'ij_x',
        'ij_y',
        'xy_x',
        'xy_y',
        'pq_x',
        'pq_y',
        'sig_x_loc',
        'sig_y_loc',
    ])
    telescope = QTable([
        env['orbit_id'],
        env['field_id'],
        env['plate_id'],
        params['tel_a_loc'] * u.degree,
        params['tel_d_loc'] * u.degree,
        params['tel_t_loc'] * u.degree,
        params['tel_s_loc'][0, :] * 180.0 / np.pi * u.um,
        params['tel_s_loc'][1, :] * 180.0 / np.pi * u.um,
    ], names=[
        'orbit_id',
        'field_id',
        'plate_id',
        'tel_a_loc',
        'tel_d_loc',
        'tel_t_loc',
        'tel_f_loc_x',
        'tel_f_loc_y',
    ])
    detector = QTable([
        params['det_r_loc'],
        params['det_o_loc'][:, 0],
        params['det_o_loc'][:, 1],
        params['det_p_loc'][:, 0],
        params['det_p_loc'][:, 1],
    ], names=[
        'det_r_loc',
        'det_o_loc_x',
        'det_o_loc_y',
        'det_p_loc_x',
        'det_p_loc_y',
    ])
    coeff = QTable([
        params['opt_A_loc'],
        params['opt_B_loc'],
    ], names=[
        'opt_A_loc',
        'opt_B_loc',
    ])
    trace = QTable([
        losses,
    ], names=[
        'loss',
    ])
    if args.verbose:
        print(result)

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=phead),
        fits.BinTableHDU(name='result', data=result, header=rhead),
        fits.BinTableHDU(name='sample', data=sample),
        fits.BinTableHDU(name='telescope', data=telescope),
        fits.BinTableHDU(name='detector', data=detector),
        fits.BinTableHDU(name='distortion', data=coeff),
        fits.BinTableHDU(name='trace', data=trace)
    ])
    hdul.writeto(args.output, overwrite=args.overwrite)
