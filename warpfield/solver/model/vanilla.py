#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' vanilla model '''

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.distributions import constraints as c
import numpyro.distributions as dist
import numpyro

from ..projection.gnomonic import projection



def generate(source, reference, params={}):
    ''' generate model and guide functions


    This function generates model and guide functions for inference
    using jax from source and reference data tables. The source table
    should contains the following columns:

      source:
        x: x-coordinates of sources on the focal plane in mm.
        y: y-coordinates of sources on the focal plane in mm.
        object_id: unique ID numbers of objects.
        plate_id: ID numbers of observation plates (pointings).

    The reference table should contains the folloinwg columns:

      reference:
        object_id: unique ID numbers of objects.
        ra: right acensions of objects in degree.
        dec: declinations of objects in degree.
        sig: positional uncertainty in degree.


    Arguments:
      source: pandas DataFrame of measurements.
      reference: pandas DataFrame of reference stars.
      params: dictionary of the initial condition.

    Returns:
      a function pair (model, guide) is generated.

        model: the obervation model function.
        guide: a guide function for SVI.

    '''
    T = source.shape[0]
    N = source.plate_id.unique().size
    M = reference.shape[0]
    F0 = 1.0 / 7.84e-6
    sigma = params.get('sigma', 4e-3/3600)

    x0 = jnp.array(source['x'])
    y0 = jnp.array(source['y'])

    tel_a_loc = jnp.array(params.get('tel_a_loc', np.tile(266.335, N)))
    tel_a_sig = jnp.array(params.get('tel_a_sig', np.tile(1.0, N)))
    tel_d_loc = jnp.array(params.get('tel_d_loc', np.tile(-28.642, N)))
    tel_d_sig = jnp.array(params.get('tel_d_sig', np.tile(1.0, N)))
    tel_t_loc = jnp.array(params.get('tel_t_loc', np.tile(58.80, N)))
    tel_t_sig = jnp.array(params.get('tel_t_sig', np.tile(1.0, N)))
    foc_f_loc = jnp.array(params.get('foc_f_loc', np.tile(1.0, [1, 2])))
    foc_f_sig = jnp.array(params.get('foc_f_sig', np.tile(0.1, [1, 2])))

    ref_a_loc = jnp.array(params.get('ra_loc', reference['ra_p']))
    ref_a_sig = jnp.array(params.get('ra_sig', reference['sig']))
    ref_d_loc = jnp.array(params.get('dec_loc', reference['dec_p']))
    ref_d_sig = jnp.array(params.get('dec_sig', reference['sig']))

    pri_a_loc = jnp.array(reference['ra_p'])
    pri_d_loc = jnp.array(reference['dec_p'])
    pri_a_sig = jnp.array(reference['sig'])
    pri_d_sig = jnp.array(reference['sig'])

    plate_id = pd.Index(source.plate_id.unique())
    object_id = pd.Index(reference.object_id)
    pidx = jnp.array([plate_id.get_loc(p) for p in source.plate_id])
    oidx = jnp.array([object_id.get_loc(o) for o in source.object_id])

    def model():
        with numpyro.plate('ref', M):
            ra = numpyro.sample('ra', dist.Normal(pri_a_loc, pri_a_sig))
            dec = numpyro.sample('dec', dist.Normal(pri_d_loc, pri_d_sig))
        with numpyro.plate('pointing', N):
            a = numpyro.sample('tel_a', dist.Normal(tel_a_loc, 1))
            d = numpyro.sample('tel_d', dist.Normal(tel_d_loc, 1))
            t = numpyro.sample('tel_t', dist.Normal(tel_t_loc, 5))
        f = numpyro.sample('foc_f', dist.Uniform(0.95, 1.05))
        F = numpyro.deterministic('foc_F0', f * F0)

        ax = jnp.take(a, jnp.array(pidx))
        dx = jnp.take(d, jnp.array(pidx))
        tx = jnp.take(t, jnp.array(pidx))
        fx = jnp.tile(F, [T, 1])

        rax = jnp.take(ra, jnp.array(oidx))
        dex = jnp.take(dec, jnp.array(oidx))

        xy = numpyro.deterministic('xy', projection(ax, dx, tx, rax, dex, fx))

        with numpyro.plate('obs', T):
            numpyro.sample('x', dist.Normal(xy[:, 0], sigma), obs=x0)
            numpyro.sample('y', dist.Normal(xy[:, 1], sigma), obs=y0)

    def guide():
        loc_a = numpyro.param('tel_a_loc',
                              tel_a_loc,
                              constraint=c.interval(tel_a_loc - 1,
                                                    tel_a_loc + 1))
        sig_a = numpyro.param('tel_a_sig', tel_a_sig, constraint=c.positive)
        loc_d = numpyro.param('tel_d_loc',
                              tel_d_loc,
                              constraint=c.interval(tel_d_loc - 1,
                                                    tel_d_loc + 1))
        sig_d = numpyro.param('tel_d_sig', tel_d_sig, constraint=c.positive)
        loc_t = numpyro.param('tel_t_loc',
                              tel_t_loc,
                              constraint=c.interval(tel_t_loc - 5,
                                                    tel_t_loc + 5))
        sig_t = numpyro.param('tel_t_sig', tel_t_sig, constraint=c.positive)
        loc_f = numpyro.param('foc_f_loc',
                              foc_f_loc,
                              constraint=c.interval(0.9, 1.1))
        sig_f = numpyro.param('foc_f_sig', foc_f_sig, constraint=c.positive)

        with numpyro.plate('pointing', N):
            numpyro.sample('tel_a', dist.Normal(loc_a, sig_a))
            numpyro.sample('tel_d', dist.Normal(loc_d, sig_d))
            numpyro.sample('tel_t', dist.Normal(loc_t, sig_t))
        numpyro.sample('foc_f', dist.Normal(loc_f, sig_f))

        with numpyro.plate('ref', M):
            loc_ra = numpyro.param('ra_loc',
                                   ref_a_loc,
                                   constraint=c.interval(
                                       ref_a_loc - 10 * ref_a_sig,
                                       ref_a_loc + 10 * ref_a_sig))
            sig_ra = numpyro.param('ra_sig', ref_a_sig, constraint=c.positive)
            loc_dec = numpyro.param('dec_loc',
                                    ref_d_loc,
                                    constraint=c.interval(
                                        ref_d_loc - 10 * ref_d_sig,
                                        ref_d_loc + 10 * ref_d_sig))
            sig_dec = numpyro.param('dec_sig',
                                    ref_d_sig,
                                    constraint=c.positive)

            numpyro.sample('ra', dist.Normal(loc_ra, sig_ra))
            numpyro.sample('dec', dist.Normal(loc_dec, sig_dec))

    return model, guide
