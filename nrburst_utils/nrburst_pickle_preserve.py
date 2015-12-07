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
nrburst_pickle_preserve.py

Crunch together pickles from nrburst_match.py

"""

import sys
import glob
import cPickle as pickle
import numpy as np

pickle_files = glob.glob(sys.argv[1]+'*pickle')
user_tag = sys.argv[2]

# Load first one to extract data for preallocation
current_matches, current_masses, current_inclinations, config, \
        simulations, __author__, __version__, __date__ = \
        pickle.load(open(pickle_files[0],'r'))

if config.algorithm=='BW':

    nSims = len(pickle_files)
    nsampls = config.nsampls

    # --- Preallocate
    matches = np.zeros(shape=(nSims, nsampls))
    masses  = np.zeros(shape=(nSims, nsampls))
    inclinations = np.zeros(shape=(nSims, nsampls))

    # be a bit careful with the simulations object
    setattr(simulations, 'simulations', [])
    setattr(simulations, 'nsimulations', nSims)

    for f, file in enumerate(pickle_files):
        current_matches, current_masses, current_inclinations, config, \
                current_simulations, __author__, __version__, __date__ = \
                pickle.load(open(file,'r'))

        matches[f,:] = current_matches[0][0]
        masses[f,:]  = current_masses[0][0]
        inclinations[f,:] = current_inclinations[0][0]

        simulations.simulations.append(current_simulations.simulations[0])

else:
    matches = current_matches[0]
    masses  = current_masses[0]
    inclinations = current_inclinations[0]

filename=user_tag+'_'+config.algorithm+'.pickle'

pickle.dump([matches, masses, inclinations, config, simulations,
        __author__, __version__, __date__], open(filename, "wb"))


