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
nrburst_matches.py

Compute matches between burst reconstruction and NR waveforms
"""

import sys, os
import subprocess
import cPickle as pickle

import numpy as np
import scipy.optimize
import timeit

import lal
from pylal import spawaveform

import nrburst_utils as nrbu

__author__ = "James Clark <james.clark@ligo.org>"
#git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
#__version__ = "git id %s" % git_version_id

gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse input

opts, args, cp = nrbu.parser()
config = nrbu.configuration(cp)

#
# --- catalog Definition
#
bounds = dict()
bounds['Mchirpmin30Hz'] = [-np.inf, config.min_chirp_mass]

#
# --- Reconstruction data
#
print >> sys.stdout,  "Loading data"
reconstruction_data = np.loadtxt(config.reconstruction)
asd_data = np.loadtxt(config.spectral_estimate)

# If BayesWave, select the user-specified number of samples for which we will
# compute matches (useful for speed / development work)
if config.algorithm=='BW':
    print 'reducing sample size'
    idx = np.random.random_integers(low=0, high=len(reconstruction_data)-1,
            size=config.nsampls)
    reconstruction_data = reconstruction_data[idx]
elif config.algorithm=='CWB':
    reconstruction_data = nrbu.extract_wave(reconstruction_data, config.datalen,
            config.sample_rate)
    # Make it iterable so that the BW/CWB codes can be consistent
    reconstruction_data = [reconstruction_data]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate The catalog

init_total_mass = 100   # Generate a catalog at this mass; shouldn't matter,
                        # we rescale anyway
distance=1. # Mpc

#
# --- Generate initial catalog
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()
simulations = \
        nrbu.simulation_details(param_bounds=bounds,
                catdir=config.catalog)

print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Building NR catalog'
print >> sys.stdout,  ''
catalog = nrbu.waveform_catalog(simulations, ref_mass=init_total_mass,
        SI_deltaT=config.deltaT, SI_datalen=config.datalen, distance=distance,
        trunc_time=False)
now = timeit.time.time()
print >> sys.stdout,  "...catalog construction took %.1f..."%(now-then)


# Useful time/freq samples
time_axis = np.arange(0, config.datalen, config.deltaT)
freq_axis = np.arange(0, catalog.SI_flen*catalog.SI_deltaF,
        catalog.SI_deltaF)

# Interpolate the ASD to the waveform frequencies (this is convenient so that we
# end up with a PSD which overs all frequencies for use in the match calculation
# later - In practice, this will really just pad out the spectrum at low
# frequencies)
asd = np.interp(freq_axis, asd_data[:,0], asd_data[:,1])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameter Estimation
#

#
# --- Minimise mismatch over waveforms and total mass
#

# Preallocate
matches = np.zeros(shape=(simulations.nsimulations, len(reconstruction_data)))
masses  = np.zeros(shape=(simulations.nsimulations, len(reconstruction_data)))

# Loop over waves in NR catalog
for w, wave in enumerate(catalog.SIComplexTimeSeries):

    print >> sys.stdout,  "________________________________"
    print >> sys.stdout,  "Computing match for %s (%d/%d)"%(simulations.simulations[w]['wavename'],
            w+1, simulations.nsimulations)


    # Find best-fitting mass (in terms of match)
    print >> sys.stdout,  "Optimising for total mass for each sampled waveform..."

    # Find min/max allowable mass to which we can scale the waveform
    min_mass = nrbu.mtot_from_mchirp(config.min_chirp_mass,
            simulations.simulations[w]['q'])
    max_mass = nrbu.mtot_from_mchirp(config.max_chirp_mass,
            simulations.simulations[w]['q'])

    mass_guess = nrbu.mtot_from_mchirp(config.mass_guess,
            simulations.simulations[w]['q']) 


    for s, sample in enumerate(reconstruction_data):


        print >> sys.stdout,  '-----------------------------'
        print >> sys.stdout,  "Evaluating sample waveform %d of %d"%( s,
                len(reconstruction_data) )
        print >> sys.stdout,  " (NR: %s [%d/%d])"%( simulations.simulations[w]['wavename'], w+1,
                simulations.nsimulations)

        #
        # Optimise match over total mass
        #

        then = timeit.time.time()
        ifo_response=True
        result = scipy.optimize.fmin(nrbu.mismatch, x0=mass_guess, args=(
            init_total_mass, [min_mass, max_mass], wave, sample, asd, config.deltaT,
            catalog.SI_deltaF, ifo_response, config.f_min), full_output=True,
            retall=True, disp=True)
        now = timeit.time.time()
        print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

        matches[w, s] = 1-result[1]
        masses[w, s] = result[0][0]

        mass1, mass2 = nrbu.component_masses(masses[w,s],
                simulations.simulations[w]['q'])

        print >> sys.stdout, \
                "Best match [Mchirp | Mtot]: %.2f [%.2f | %.2f]"%(
                        matches[w,s], masses[w,s], spawaveform.chirpmass(mass1,
                            mass2) / lal.MTSUN_SI)

    bestidx=np.argmax(matches[w, :])

    mass1, mass2 = nrbu.component_masses(masses[w,bestidx],
            simulations.simulations[w]['q'])

    print >> sys.stdout,  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print >> sys.stdout,  \
            " Best match [Mchirp | Mtot]: %.2f [%.2f | %.2f]"%(
            max(matches[w,:]), masses[w,bestidx], spawaveform.chirpmass(mass1,
                mass2) / lal.MTSUN_SI)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dump data

filename=opts.user_tag+'_'+config.algorithm+'_'+gpsnow+'.pickle'

#np.savez(filename, matches=matches, masses=masses)

# Dump results and configuration to pickle
pickle.dump([matches, masses, config, simulations, __author__,
    __date__], open(filename, "wb"))
#pickle.dump([matches, masses, config, simulations, __author__, __version__,
#    __date__], open(filename, "wb"))



