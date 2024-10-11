#!/usr/bin/env python
# coding: utf-8
import jax
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

import numpy as np

from dataclasses import dataclass

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

jax.config.update('jax_enable_x64', True)


__det_rotation__ = np.array(
    [0.0, -90.0, 180.0, 90.0]
)
__det_offset__ = np.array([
    [-11100.0, -11100.0],
    [+11100.0, -11100.0],
    [+11100.0, +11100.0],
    [-11100.0, +11100.0],
])
__det_pixel_size__ = np.array([
    [10.0, 10.0],
    [10.0, 10.0],
    [10.0, 10.0],
    [10.0, 10.0],
])


@dataclass
class Prior:
    src_a_loc: jax.Array
    src_a_sig: jax.Array
    src_d_loc: jax.Array
    src_d_sig: jax.Array
    tel_a_loc: jax.Array
    tel_a_sig: jax.Array
    tel_d_loc: jax.Array
    tel_d_sig: jax.Array
    tel_t_loc: jax.Array
    tel_t_sig: jax.Array
    tel_s_loc: jax.Array
    tel_s_sig: jax.Array
    opt_A_loc: jax.Array
    opt_A_sig: jax.Array
    opt_B_loc: jax.Array
    opt_B_sig: jax.Array
    det_r_loc: jax.Array
    det_r_sig: jax.Array
    det_o_loc: jax.Array
    det_o_sig: jax.Array
    det_p_loc: jax.Array
    det_p_sig: jax.Array
    sig_x_loc: jax.Array
    sig_x_sig: jax.Array
    sig_y_loc: jax.Array
    sig_y_sig: jax.Array

    @property
    def num_plate(self):
        return len(self.tel_a_loc)

    def deviation(self, name, array):
        delta = array - getattr(self, f'{name}_loc')
        return delta / getattr(self, f'{name}_sig')

    def _distribution(self, name):
        return dist.Normal(
            jnp.array(getattr(self, f'{name}_loc')),
            jnp.array(getattr(self, f'{name}_sig')))

    def __getattr__(self, name: str):
        if name == 'tel_t_dist':
            return dist.Uniform(-180, 180).expand([self.num_plate, ])
        elif name == 'tel_s_dist':
            beta = jnp.array(self.tel_s_loc / self.tel_s_sig)
            alpha = jnp.array(self.tel_s_loc * beta)
            return dist.Gamma(alpha, beta)
        elif name == 'det_p_dist':
            beta = jnp.array(self.det_p_loc / self.det_p_sig)
            alpha = jnp.array(self.det_p_loc * beta)
            return dist.Gamma(alpha, beta)
        elif name == 'sig_x_dist':
            beta = jnp.array(self.sig_x_loc / self.sig_x_sig)
            alpha = jnp.array(self.sig_x_loc * beta)
            return dist.Gamma(alpha, beta)
        elif name == 'sig_y_dist':
            beta = jnp.array(self.sig_y_loc / self.sig_y_sig)
            alpha = jnp.array(self.sig_y_loc * beta)
            return dist.Gamma(alpha, beta)
        elif name.endswith('_dist'):
            return self._distribution(name.replace('_dist', ''))
        else:
            raise NameError(f'{name} is not defined.')


def compile_prior(params, src, env, ref, num_param=18):
    f0 = env['focal length'].mean().to_value('um')

    # priors for source positions
    src_a_loc = params.get('ref_a_loc', ref['ra'].to_value('degree'))
    src_a_sig = params.get('ref_a_sig', ref['ra_error'].to_value('degree'))
    src_d_loc = params.get('ref_d_loc', ref['dec'].to_value('degree'))
    src_d_sig = params.get('ref_d_sig', ref['dec_error'].to_value('degree'))

    # priors for telescope parmeters
    tel_a_loc = env['ra est'].to_value('degree')
    tel_a_sig = 1.0 * np.ones_like(tel_a_loc)
    tel_d_loc = env['dec est'].to_value('degree')
    tel_d_sig = 1.0 * np.ones_like(tel_d_loc)
    tel_t_loc = env['pa est'].to_value('degree')
    tel_t_sig = 1.0 * np.ones_like(tel_t_loc)
    tel_s_loc = np.tile(f0, [2, len(tel_a_loc)]) / 180.0 * jnp.pi
    tel_s_sig = 100.0 * np.ones_like(tel_s_loc)

    # priors for optics distortion
    opt_A_loc = 0.0 * np.ones(num_param)
    opt_A_sig = 1.0 * np.ones(num_param)
    opt_B_loc = 0.0 * np.ones(num_param)
    opt_B_sig = 1.0 * np.ones(num_param)

    # priors for detector arrangement
    det_r_loc = __det_rotation__
    det_r_sig = 0.1 * np.ones_like(det_r_loc)
    det_o_loc = __det_offset__
    det_o_sig = 100.0 * np.ones_like(det_o_loc)
    det_p_loc = __det_pixel_size__
    det_p_sig = 0.001 * np.ones_like(det_p_loc)

    # priors for measurement errors
    sig_x_loc = src['sx0'].to_value('pixel')
    sig_x_sig = src['sx0'].to_value('pixel') * 0.01
    sig_y_loc = src['sy0'].to_value('pixel')
    sig_y_sig = src['sy0'].to_value('pixel') * 0.01

    return Prior(
        src_a_loc=src_a_loc, src_a_sig=src_a_sig,
        src_d_loc=src_d_loc, src_d_sig=src_d_sig,
        tel_a_loc=tel_a_loc, tel_a_sig=tel_a_sig,
        tel_d_loc=tel_d_loc, tel_d_sig=tel_d_sig,
        tel_t_loc=tel_t_loc, tel_t_sig=tel_t_sig,
        tel_s_loc=tel_s_loc, tel_s_sig=tel_s_sig,
        opt_A_loc=opt_A_loc, opt_A_sig=opt_A_sig,
        opt_B_loc=opt_B_loc, opt_B_sig=opt_B_sig,
        det_r_loc=det_r_loc, det_r_sig=det_r_sig,
        det_o_loc=det_o_loc, det_o_sig=det_o_sig,
        det_p_loc=det_p_loc, det_p_sig=det_p_sig,
        sig_x_loc=sig_x_loc, sig_x_sig=sig_x_sig,
        sig_y_loc=sig_y_loc, sig_y_sig=sig_y_sig,
    )


@dataclass
class InitialValue:
    src_a_loc: jax.Array
    src_a_sig: jax.Array
    src_d_loc: jax.Array
    src_d_sig: jax.Array
    tel_a_loc: jax.Array
    tel_d_loc: jax.Array
    tel_t_loc: jax.Array
    tel_s_loc: jax.Array
    opt_A_loc: jax.Array
    opt_B_loc: jax.Array
    det_r_loc: jax.Array
    det_o_loc: jax.Array
    det_p_loc: jax.Array
    sig_x_loc: jax.Array
    sig_y_loc: jax.Array

    def get(self, name):
        return jnp.array(getattr(self, name))

    def param(self, name, s=1.0, **options):
        return numpyro.param(name, self.get(name), **options) / s

    def value(self, name):
        return self.get(name)


def compile_initial_value(params, src, env, ref, num_param=18):
    f0 = env['focal length'].mean().to_value('um')

    # source positions
    src_a_loc = params.get('src_a_loc', ref['ra'].to_value('degree'))
    src_a_sig = params.get('src_a_sig', ref['ra_error'].to_value('degree'))
    src_d_loc = params.get('src_d_loc', ref['dec'].to_value('degree'))
    src_d_sig = params.get('src_d_sig', ref['dec_error'].to_value('degree'))

    # telescope parameters
    tel_a_loc = params.get('tel_a_loc', env['ra est'].to_value('degree'))
    tel_d_loc = params.get('tel_d_loc', env['dec est'].to_value('degree'))
    tel_t_loc = params.get('tel_t_loc', env['pa est'].to_value('degree'))
    tel_f_loc = params.get('tel_f_loc', np.tile(f0, [len(tel_a_loc), 2]))
    tel_s_loc = tel_f_loc.T / 180.0 * jnp.pi

    # optical distortion
    opt_A_loc = params.get('opt_A_loc', 0.0 * np.ones(num_param))
    opt_B_loc = params.get('opt_B_loc', 0.0 * np.ones(num_param))

    # detector arrangement
    det_r_loc = params.get('det_r_loc', __det_rotation__)
    det_o_loc = params.get('det_o_loc', __det_offset__)
    det_p_loc = params.get('det_p_loc', __det_pixel_size__)

    # measurement error
    sig_x_loc = params.get('sigma_loc', src['sx0'].to_value('pixel'))
    sig_y_loc = params.get('sigma_loc', src['sy0'].to_value('pixel'))

    return InitialValue(
        src_a_loc=src_a_loc, src_a_sig=src_a_sig,
        src_d_loc=src_d_loc, src_d_sig=src_d_sig,
        tel_a_loc=tel_a_loc, tel_d_loc=tel_d_loc,
        tel_t_loc=tel_t_loc, tel_s_loc=tel_s_loc,
        opt_A_loc=opt_A_loc, opt_B_loc=opt_B_loc,
        det_r_loc=det_r_loc, det_o_loc=det_o_loc, det_p_loc=det_p_loc,
        sig_x_loc=sig_x_loc, sig_y_loc=sig_y_loc,
    )
