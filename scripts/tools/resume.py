#!/usr/bin/env python
# coding: utf-8
from astropy.table import QTable
import numpy as np


def setup_params(params, resume):
    r = QTable.read(resume, hdu='result')
    s = QTable.read(resume, hdu='sample')
    t = QTable.read(resume, hdu='telescope')
    d = QTable.read(resume, hdu='distortion')
    c = QTable.read(resume, hdu='detector')
    params['src_a_loc'] = r['src_a_loc']
    params['src_a_sig'] = r['src_a_sig']
    params['src_d_loc'] = r['src_d_loc']
    params['src_d_sig'] = r['src_d_sig']
    params['tel_a_loc'] = t['tel_a_loc']
    params['tel_d_loc'] = t['tel_d_loc']
    params['tel_t_loc'] = t['tel_t_loc']
    params['tel_a_loc'] = t['tel_a_loc']
    params['tel_f_loc_x'] = t['tel_f_loc_x']
    params['tel_f_loc_y'] = t['tel_f_loc_y']
    params['tel_f_loc'] = np.stack([t['tel_f_loc_x'], t['tel_f_loc_y']]).T
    params['opt_A_loc'] = d['opt_A_loc']
    params['opt_B_loc'] = d['opt_B_loc']
    params['det_r_loc'] = c['det_r_loc']
    params['det_o_loc'] = np.stack([c['det_o_loc_x'], c['det_o_loc_y']]).T
    params['det_p_loc'] = np.stack([c['det_p_loc_x'], c['det_p_loc_y']]).T
    params['sig_x_loc'] = s['sig_x_loc']
    params['sig_y_loc'] = s['sig_y_loc']
    return params
