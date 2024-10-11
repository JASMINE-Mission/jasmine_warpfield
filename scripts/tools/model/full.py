#!/usr/bin/env python
# coding: utf-8
import jax
import numpyro

from numpyro.distributions import constraints as c
import numpyro.distributions as dist
import jax.numpy as jnp

from astropy.table import unique

from warpfield.analysis.distortion.legendre import distortion
from warpfield.analysis.projection.gnomonic import projection
from warpfield.analysis.transform.affine import transform

from ..propagate import propagate
from ..compile import compile_prior, compile_initial_value

jax.config.update('jax_enable_x64', True)


def generate(src, env, ref, params={}):
    ''' Generate model and guide functions

    Parameters:
        src (QTable):
          A QTable containing measured positions on the detector coordinates.
        env (QTable):
          A QTable containing observation environments.
        ref (QTable):
          A QTable containing reference catalog.
    '''
    obstime = params['obstime']
    ref = propagate(ref, obstime)

    m = len(unique(src, 'plate_id'))
    src['plate_seq'] = src['field_id'] * m + src['plate_id']
    plate = unique(src, ['plate_seq'])['plate_seq', ]
    det = unique(src, 'detector_id')['detector_id', ]

    plane_scale = 2.1e4

    prior = compile_prior(params, src, env, ref)
    init = compile_initial_value(params, src, env, ref)

    nx = jnp.array(src['nx'])
    ny = jnp.array(src['ny'])
    r = jnp.array(ref['reference_flag'])

    plate.add_index('plate_seq')
    det.add_index('detector_id')
    ref.add_index('source_id')
    pidx = jnp.array(plate.loc_indices[src['plate_seq']])
    sidx = jnp.array(ref.loc_indices[src['source_id']])
    didx = jnp.array(det.loc_indices[src['detector_id']])

    def model():
        src_a = numpyro.sample('ra',  prior.src_a_dist)
        src_d = numpyro.sample('dec', prior.src_d_dist)

        tel_a = numpyro.sample('tel_a', prior.tel_a_dist)
        tel_d = numpyro.sample('tel_d', prior.tel_d_dist)
        tel_t = numpyro.sample('tel_t', prior.tel_t_dist)
        tel_s = numpyro.sample('tel_s', prior.tel_s_dist)

        opt_A = numpyro.sample('opt_A', prior.opt_A_dist)
        opt_B = numpyro.sample('opt_B', prior.opt_B_dist)

        ax = jnp.take(tel_a, pidx)
        dx = jnp.take(tel_d, pidx)
        tx = jnp.take(tel_t, pidx)
        sx = jnp.take(tel_s, pidx, axis=1).T

        rax = jnp.take(src_a, sidx)
        dex = jnp.take(src_d, sidx)

        det_r = numpyro.sample('det_r', prior.det_r_dist)
        det_o = numpyro.sample('det_o', prior.det_o_dist)
        det_p = numpyro.sample('det_p', prior.det_p_dist)

        rx = jnp.take(det_r, didx, axis=0)
        ox = jnp.take(det_o, didx, axis=0)
        px = jnp.take(det_p, didx, axis=0)

        pq = numpyro.deterministic(
            'pq', projection(ax, dx, tx, rax, dex, sx))
        xy = numpyro.deterministic(
            'xy', pq + distortion(opt_A, opt_B, pq / plane_scale))
        ij = numpyro.deterministic(
            'ij', transform(xy, rx, ox, px))

        sig_x = init.value('sig_x_loc')
        sig_y = init.value('sig_y_loc')

        numpyro.deterministic(
            'sigma_a', jnp.std(prior.deviation('src_a', src_a)[r]))
        numpyro.deterministic(
            'sigma_d', jnp.std(prior.deviation('src_d', src_d)[r]))

        numpyro.sample('nx', dist.Normal(ij[:, 0], sig_x), obs=nx)
        numpyro.sample('ny', dist.Normal(ij[:, 1], sig_y), obs=ny)

    def guide():
        a_loc = init.param('tel_a_loc')
        d_loc = init.param('tel_d_loc')
        t_loc = init.param('tel_t_loc')
        s_loc = init.param('tel_s_loc', constraint=c.positive)

        numpyro.sample('tel_a', dist.Delta(a_loc))
        numpyro.sample('tel_d', dist.Delta(d_loc))
        numpyro.sample('tel_t', dist.Delta(t_loc))
        numpyro.sample('tel_s', dist.Delta(s_loc))

        opt_A_loc = init.param('opt_A_loc')
        opt_B_loc = init.param('opt_B_loc')
        numpyro.sample('opt_A', dist.Delta(opt_A_loc))
        numpyro.sample('opt_B', dist.Delta(opt_B_loc))

        det_r_loc = init.param('det_r_loc')
        det_o_loc = init.param('det_o_loc')
        det_p_loc = init.param('det_p_loc', constraint=c.positive)
        numpyro.sample('det_r', dist.Delta(det_r_loc))
        numpyro.sample('det_o', dist.Delta(det_o_loc))
        numpyro.sample('det_p', dist.Delta(det_p_loc))

        init.param('sig_x_loc', constraint=c.positive)
        init.param('sig_y_loc', constraint=c.positive)

        src_a_loc  = init.param('src_a_loc')
        src_a_sig  = init.param('src_a_sig', constraint=c.positive)
        src_d_loc = init.param('src_d_loc')
        src_d_sig = init.param('src_d_sig', constraint=c.positive)

        numpyro.sample('ra', dist.Normal(src_a_loc, src_a_sig))
        numpyro.sample('dec', dist.Normal(src_d_loc, src_d_sig))

    return model, guide
