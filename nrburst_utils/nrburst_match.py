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
nrburst_match.py

Compute matches between burst reconstruction and NR waveforms
"""

import sys, os
import os.path
import subprocess
import cPickle as pickle

import numpy as np
import scipy.optimize
import timeit

import lal
from pylal import spawaveform

import nrburst_utils as nrbu

__author__ = "James Clark <james.clark@ligo.org>"
gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# Get the current git version
git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(sys.argv[0])).strip()
__version__ = "git id %s" % git_version_id


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
#    reconstruction_data = reconstruction_data[idx]
elif config.algorithm=='CWB':
    reconstruction_data = nrbu.extract_wave(reconstruction_data, config.datalen,
            config.sample_rate)
    # Make it iterable so that the BW/CWB codes can be consistent
    reconstruction_data = [reconstruction_data]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate The catalog

#
# --- Generate initial catalog
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()

simulations = nrbu.simulation_details(param_bounds=bounds,
        catdir=config.catalog)

# Useful time/freq samples
time_axis = np.arange(config.datalen, config.delta_t)
freq_axis = np.arange(0.5*config.datalen/config.delta_t+1./config.datalen) * 1./config.datalen

# Interpolate the ASD to the waveform frequencies (this is convenient so that we
# end up with a PSD which overs all frequencies for use in the match calculation
# later - In practice, this will really just pad out the spectrum at low
# frequencies)
asd = np.interp(freq_axis, asd_data[:,0], asd_data[:,1])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameter Estimation
#

#
# --- Compute Fitting-Factor for each NR waveform
#
# Fitting factor: normalised inner product, maximised over time, phase-offset,
# total mass and orientation

# Preallocate
matches = np.zeros(shape=(simulations.nsimulations, len(reconstruction_data)))
masses  = np.zeros(shape=(simulations.nsimulations, len(reconstruction_data)))
inclinations  = np.zeros(shape=(simulations.nsimulations, len(reconstruction_data)))
polarizations = np.zeros(shape=(simulations.nsimulations, len(reconstruction_data)))

# ----------------------------------------------------------------------------------
# XXX: Debugging
inc = 90*np.random.random()
pol = 0*np.random.random()
mass = 74.

print "INJECTING:"
print mass, inc, pol

# Generate the polarisations
hp, hc = nrbu.get_wf_pols(simulations.simulations[0]['wavefile'], mass,
        inclination=inc, delta_t=config.delta_t)

# Project to detector
data = nrbu.project_waveform(hp, hc, skyloc=(0,0),
        polarization=pol, detector_name=config.detector_name)

# Resize to the same length as the data
data.resize(config.datalen*config.sample_rate)

# Whiten the template
Data = data.to_frequencyseries()
Data.data /= asd 

data = Data.to_timeseries()
# ----------------------------------------------------------------------------------


# Loop over waves in NR catalog
for w in xrange(simulations.nsimulations):

    print >> sys.stdout,  "________________________________"
    print >> sys.stdout,  "Computing match (%d/%d)"%( w+1,
            simulations.nsimulations)

    # Find best-fitting mass (in terms of match)
    print >> sys.stdout,  "Optimising for total mass for each sampled waveform..."

    # Find min/max allowable mass to which we can scale the waveform
    min_mass = nrbu.mtot_from_mchirp(config.min_chirp_mass,
            simulations.simulations[w]['q'])
    max_mass = nrbu.mtot_from_mchirp(config.max_chirp_mass,
            simulations.simulations[w]['q'])

    # Starting point for param maximisation
    mass_guess = nrbu.mtot_from_mchirp(config.mass_guess,
            simulations.simulations[w]['q']) 
    mass_guess = (max_mass - min_mass)*np.random.random() + min_mass 

    inclination_guess  = 90*np.random.random()
    polarization_guess = 360*np.random.random()

    init_guess = np.array([mass_guess, inclination_guess, polarization_guess])
    #init_guess = mass_guess + -2*np.random.random()

    print "INITAL GUESS:"
    print init_guess

    for s, sample in enumerate(reconstruction_data):

        sample=data.data[:]

        print >> sys.stdout,  '-----------------------------'
        print >> sys.stdout,  "Evaluating sample waveform %d of %d"%( s,
                len(reconstruction_data) )
        print >> sys.stdout,  " (NR: [%d/%d])"%(w+1, simulations.nsimulations)

        #
        # Optimise match over total mass
        #
        then = timeit.time.time()

        result = scipy.optimize.fmin(nrbu.mismatch,
                x0=init_guess,
                args=(
                    (0,0), simulations.simulations[w]['wavefile'],
                    config.detector_name,
                    (min_mass, max_mass),
                    sample, asd, config.delta_t
                    ), xtol=1e-10, ftol=1e-10, maxfun=10000,
                full_output=True, retall=True, disp=True)

        now = timeit.time.time()
        print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

        matches[w, s] = 1-result[1]
        masses[w, s]  = result[0][0]
        inclinations[w, s]  = result[0][1]
        polarizations[w, s] = result[0][2]

        mass1, mass2 = nrbu.component_masses(masses[w,s],
                simulations.simulations[w]['q'])

        print >> sys.stdout, ""
        print >> sys.stdout, "Fit-factor: %.2f"%(matches[w,s])
        print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f"%(
                spawaveform.chirpmass(mass1, mass2) / lal.MTSUN_SI, masses[w,s])
        print >> sys.stdout, "inclination=%.2f, polarization=%.2f"%(
                inclinations[w, s], polarizations[w, s])

        print "%f %f %f"%(masses[w, s], inclinations[w, s], polarizations[w, s])

        # ----------------------------------------------------------------------------------
        # XXX: DEBUGGING

        from matplotlib import pyplot as pl
        import pycbc.types

        # Generate the polarisations
        hp, hc = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'], masses[w,s],
                inclination=inclinations[w,s], delta_t=config.delta_t)

        # Project to detector
        tmplt = nrbu.project_waveform(hp, hc, skyloc=(0,0),
                polarization=polarizations[w,s],
                detector_name=config.detector_name)

        # Resize to the same length as the data
        tlen = max(len(tmplt), len(sample))
        tmplt.resize(tlen)

        # Whiten the template
        Tmplt = tmplt.to_frequencyseries()
        Tmplt.data /= asd 

        tmplt = Tmplt.to_timeseries()
        tmplt.data /= pycbc.filter.sigma(tmplt)

        data.data /= pycbc.filter.sigma(data)

        pl.figure()
        pl.plot(tmplt.sample_times - tmplt.sample_times[np.argmax(tmplt)],
                tmplt, label='tmplt')
        pl.plot(data.sample_times - data.sample_times[np.argmax(data)],
                data, label='data')
        pl.legend()
        pl.title('%f'%matches[w,s])
        pl.xlim(-0.15,0.1)
        pl.show()

        # ----------------------------------------------------------------------------------

        sys.exit()

    bestidx=np.argmax(matches[w, :])

    mass1, mass2 = nrbu.component_masses(masses[w,bestidx],
            simulations.simulations[w]['q'])

    print >> sys.stdout, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print >> sys.stdout, "Best Match:"

    print >> sys.stdout, "Fit-factor: %.2f"%(matches[w,bestidx])
    print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f"%(
            spawaveform.chirpmass(mass1, mass2) / lal.MTSUN_SI, masses[w,bestidx])
    print >> sys.stdout, "inclination=%.2f, polarization=%.2f"%(
            inclinations[w, bestidx], polarizations[w, bestidx])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dump data

filename=opts.user_tag+'_'+config.algorithm+'_'+gpsnow+'.pickle'

# Dump results and configuration to pickle
pickle.dump([matches, masses, inclinations, polarizations, config, simulations,
    __author__, __version__, __date__], open(filename, "wb"))



