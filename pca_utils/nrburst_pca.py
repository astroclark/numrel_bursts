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
bhex_pca.py
"""

import sys, os
import os.path
import subprocess
import cPickle as pickle

import timeit
import numpy as np

import nrburst_pca_utils as nrbu_pca
import nrburst_utils as nrbu

#
# --- catalog Definition
#
bounds = dict()
bounds['Mchirpmin30Hz'] = [-np.inf, 30.0]
bounds['a1'] = [-0.99, 0.01]
bounds['a2'] = [-0.99, 0.01]

noise_file = \
        '/home/jclark/Projects/bhextractor/data/noise_curves/early_aligo.dat'

#
# --- Generate initial catalog
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()

# Select simulations
simulations = nrbu.simulation_details(param_bounds=bounds,
        catdir='/data/lvc_nr/GaTech')

# Build catalog from HDF5
catalog = nrbu_pca.catalog(simulations)

# Peform PCA
bbh_pca = nrbu_pca.bbh_pca(catalog, noise_file=noise_file)

# Save PCA data
bbh_pca.file_dump(sys.argv[1])



