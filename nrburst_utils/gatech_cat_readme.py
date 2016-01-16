#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
gatech_cat_readme.py

Parse hdf5 files in gatech directory and produce a readme file for easy plotting
and bespoke catalogue creation (e.g., non-spinning)

"""

from __future__ import division

import os
import sys 
import glob

import numpy as np

import h5py

from pycbc import pnutils

import nrburst_utils as nrbu


def get_params(file):
    """
    Read waveform meta data from hdf5 file
    """

    f = h5py.File(file, 'r')

    # Metadata parameters:
    params = {}

    params['eta'] = float(f.attrs['eta'])

    params['spin1x'] = float(f.attrs['spin1x'])
    params['spin1y'] = float(f.attrs['spin1y'])
    params['spin1z'] = float(f.attrs['spin1z'])
    params['spin2x'] = float(f.attrs['spin2x'])
    params['spin2y'] = float(f.attrs['spin2y'])
    params['spin2z'] = float(f.attrs['spin2z'])

    params['Mmin30Hz'] = float(f.attrs['f_lower_at_1MSUN']) / 30.0

    mass1, mass2 = pnutils.mtotal_eta_to_mass1_mass2(params['Mmin30Hz'],
            params['eta'])

    params['Mchirpmin30Hz'], _ = \
            pnutils.mass1_mass2_to_mchirp_eta(mass1,mass2)

    params['q'] = mass1 / mass2

    # --- Derived spin configuration
    params['a1'] = np.linalg.norm([params['spin1x'], params['spin1y'], params['spin1z']])
    params['a2'] = np.linalg.norm([params['spin2x'], params['spin2y'], params['spin2z']])

    params['a1dotL'], vec =  nrbu.a_with_L(params['spin1x'], params['spin1y'],
            params['spin1z'])
    params['a1crossL'] = np.linalg.norm(vec)

    params['a2dotL'], vec =  nrbu.a_with_L(params['spin2x'], params['spin2y'],
            params['spin2z'])
    params['a2crossL'] = np.linalg.norm(vec)

    params['theta_a12'] = nrbu.spin_angle(params['spin1x'], params['spin1y'], params['spin1z'], 
            params['spin2x'], params['spin2y'], params['spin2z'])

    params['SeffdotL'], vec = nrbu.effspin_with_L(params['q'], 
                        params['spin1x'], params['spin1y'],
                        params['spin1z'], params['spin2x'], params['spin2y'],
                        params['spin2z'])
    params['SeffcrossL'] = np.linalg.norm(vec)

    params['SdotL'], params['theta_SdotL'] = nrbu.totspin_dot_L(params['q'], 
                        params['spin1x'], params['spin1y'], params['spin1z'],
                        params['spin2x'], params['spin2y'], params['spin2z']) 


    params['wavefile'] = os.path.abspath(file)

    return params

#
# Identify and loop over files
#
h5files = glob.glob(sys.argv[1]+"*h5")

param_list = []
for h5file in h5files:
    param_list.append(get_params(h5file))


#
# Now write the readme
#
header = '# runID wavefile'
keys = list(np.sort(param_list[0].keys()))
keys.remove('wavefile')

for i in xrange(len(keys)):
    header += ' %s'%keys[i]

f = open('README.txt','w')
f.writelines(header+'\n')
for p,param_set in enumerate(param_list):
    line = '%d %s '%(p+1, param_set['wavefile'])
    for key in keys:
        if key=='wavefile':
            line+='%s '%(param_set[key])
        else:
            line+='%f '%(param_set[key])
    f.writelines(line+'\n')

f.close()





