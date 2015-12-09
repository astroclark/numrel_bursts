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

delta_samp=100
sample_pairs=zip(range(0,1000,delta_samp), range(delta_samp-1,1000,delta_samp))

# Get numbers for pre-allocation
sim_instances = [name.split('-')[1] for name in pickle_files]
sim_names = np.unique(sim_instances)
# XXX: assume all sample runs have same number of jobs..
n_sample_runs = sim_instances.count(sim_names[0])

# Load first one to extract data for preallocation
current_matches, current_masses, current_inclinations, config, \
        simulations, __author__, __version__, __date__ = \
        pickle.load(open(pickle_files[0],'r'))


if config.algorithm=='BW':

    nSims = len(sim_names)
    nsampls = config.nsampls * n_sample_runs

    # --- Preallocate
    matches = np.zeros(shape=(nSims, nsampls))
    masses  = np.zeros(shape=(nSims, nsampls))
    inclinations = np.zeros(shape=(nSims, nsampls))

    # be a bit careful with the simulations object
    setattr(simulations, 'simulations', [])
    setattr(simulations, 'nsimulations', nSims)

    for f, name in enumerate(sim_names):

        startidx=0
        endidx=len(current_matches[0])

        for s in xrange(n_sample_runs):

            if n_sample_runs>1:
                file = glob.glob('*%s-minsamp_%d-maxsamp_%d*'%(
                    name, min(sample_pairs[s]), max(sample_pairs[s])))[0]
            else:
                file = pickle_files[f]


            current_matches, current_masses, current_inclinations, config, \
                    current_simulations, __author__, __version__, __date__ = \
                    pickle.load(open(file,'r'))
 
            matches[f,startidx:endidx] = current_matches[0]
            masses[f,startidx:endidx] = current_masses[0]
            inclinations[f,startidx:endidx] = current_inclinations[0]
 
            startidx += len(current_matches[0])
            endidx = startidx + len(current_matches[0])

        simulations.simulations.append(current_simulations.simulations[0])

else:

    print "WHY DO YOU NEED TO DO THIS STEP FOR NON-BW RESULTS??"
    sys.exit()
    matches = current_matches[0]
    masses  = current_masses[0]
    inclinations = current_inclinations[0]

filename=user_tag+'_'+config.algorithm+'.pickle'

pickle.dump([matches, masses, inclinations, config, simulations,
        __author__, __version__, __date__], open(filename, "wb"))


