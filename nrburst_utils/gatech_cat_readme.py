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
import subprocess
import glob

import numpy as np

import h5py

from pycbc import pnutils

__author__ = "James Clark <james.clark@ligo.org>"
gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# Get the current git version
git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(sys.argv[0])).strip()
__version__ = "git id %s" % git_version_id


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

    params['a1'] = np.sqrt(params['spin1x']**2 + params['spin1y']**2 +
            params['spin1z']**2)
    params['a2'] = np.sqrt(params['spin2x']**2 + params['spin2y']**2 +
            params['spin2z']**2)

    params['Mmin30Hz'] = float(f.attrs['f_lower_at_1MSUN']) / 30.0

    mass1, mass2 = pnutils.mtotal_eta_to_mass1_mass2(params['Mmin30Hz'],
            params['eta'])

    params['Mchirpmin30Hz'], _ = \
            pnutils.mass1_mass2_to_mchirp_eta(mass1,mass2)

    params['q'] = mass1 / mass2

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





