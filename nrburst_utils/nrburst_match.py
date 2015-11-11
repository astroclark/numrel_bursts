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

    # Load sampled waveforms
    print 'reducing sample size'
    idx = np.random.random_integers(low=0, high=len(reconstruction_data)-1,
            size=config.nsampls)

    # Load extrinsic parameters
    rec_ext_params = np.loadtxt(config.extrinsic_params)
    rec_right_ascension = rec_ext_params[:,2] / lal.PI_180
    rec_declination     = np.arcsin(rec_ext_params[:,3]) / lal.PI_180
    rec_polarization    = rec_ext_params[:,4] / lal.PI_180
    
    reconstruction_data = reconstruction_data[idx]
    rec_right_ascension = rec_right_ascension[idx]
    rec_declination     = rec_declination[idx]
    rec_polarization    = rec_polarization[idx]


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
matches = np.zeros(shape=(simulations.nsimulations, config.nsampls))
masses  = np.zeros(shape=(simulations.nsimulations, config.nsampls))
inclinations  = np.zeros(shape=(simulations.nsimulations, config.nsampls))


# Loop over waves in NR catalog
for w in xrange(simulations.nsimulations):

    print >> sys.stdout,  "________________________________"
    print >> sys.stdout,  "Computing match (%d/%d)"%( w+1,
            simulations.nsimulations)

    # Find best-fitting mass (in terms of match)
    print >> sys.stdout,  "Optimising for total mass for each sampled waveform..."

    # Find min/max allowable mass to which we can scale the waveform
    min_mass = config.min_chirp_mass * simulations.simulations[w]['eta']**(-3./5.)
    max_mass = config.max_chirp_mass * simulations.simulations[w]['eta']**(-3./5.)


    # Check we can generate the polarisations
    mass_guess = (max_mass - min_mass)*np.random.random() + min_mass 
    inclination_guess  = 90*np.random.random()
    try:
        hp, hc = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'],
                mass_guess, inclination=inclination_guess, delta_t=config.delta_t)
    except:
        print >> sys.stderr, "Polarisation extraction failure, skipping %s"%(
                simulations.simulations[w]['wavefile'])
        continue


    for s, sampled_waveform in enumerate(reconstruction_data):


        # START XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXX: Debugging / Testing
        #
        # Here, we generate a pure-NR waveform and find the best-fitting parameters
        # (best fit-factor) to verify that we recover the correct parameters and match
        # when the model matches the data

#       inc = 90*np.random.random()
#       chirp_mass = config.min_chirp_mass + \
#               (config.max_chirp_mass-config.min_chirp_mass)*np.random.random()
#       mass = chirp_mass * simulations.simulations[w]['eta']**(-3./5.)
#
#       # Generate the polarisations
#       hp, hc = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'], mass,
#               inclination=inc, delta_t=config.delta_t)
#
#       # Project to detector
#       data = nrbu.project_waveform(hp, hc, skyloc=(rec_right_ascension[s], rec_declination[s]),
#               polarization=rec_polarization[s], detector_name=config.detector_name)
#
#       # Resize to the same length as the data
#       data.resize(config.datalen*config.sample_rate)
#
#       # Whiten the template
#       Data = data.to_frequencyseries()
#       Data.data /= asd 
#
#       sampled_waveform = Data.to_timeseries().data[:]
#
#       print "INJECTING:"
#       print mass, inc, rec_polarization[s]
#

        # END XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


        print >> sys.stdout, '-----------------------------'
        print >> sys.stdout, "Evaluating sample waveform %d of %d"%( s,
                len(reconstruction_data) )
        print >> sys.stdout, " NR waveform: %d/%d"%(w+1, simulations.nsimulations)
        print >> sys.stdout, " q=%.2f, a1=%.2f, a2=%.2f"%(
                simulations.simulations[w]['q'],
                simulations.simulations[w]['a1'],
                simulations.simulations[w]['a2'])

        #
        # Optimise match over total mass
        #

        # --- Starting point for param maximisation
        mass_guess = (max_mass - min_mass)*np.random.random() + min_mass 
        inclination_guess  = 90*np.random.random()
        init_guess = np.array([mass_guess, inclination_guess])

        print "INITAL GUESS:"
        print init_guess

        then = timeit.time.time()

        result = scipy.optimize.fmin(nrbu.mismatch,
                x0=init_guess,
                args=(
                    (rec_right_ascension[s], rec_declination[s]), 
                    rec_polarization[s],
                    simulations.simulations[w]['wavefile'],
                    config.detector_name,
                    (min_mass, max_mass),
                    sampled_waveform, asd, config.delta_t
                    ), xtol=1e-3, ftol=1e-3, maxfun=10000,
                full_output=True, retall=True, disp=True)

        now = timeit.time.time()
        print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

        matches[w,s] = 1-result[1]
        masses[w,s]  = result[0][0]
        inclinations[w,s]  = result[0][1]

        chirp_mass = masses[w,s]*simulations.simulations[w]['eta']**(3./5.)

        print >> sys.stdout, ""
        print >> sys.stdout, "Fit-factor: %.2f"%(matches[w,s])
        print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
                chirp_mass, masses[w,s], inclinations[w, s])

        # START XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXX: Debugging / Testing

#       from matplotlib import pyplot as pl
#       import pycbc.types
#
#       # Generate the polarisations which correspond to the best-fit parameters
#       hp, hc = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'], masses[w,s],
#               inclination=inclinations[w,s], delta_t=config.delta_t)
#
#       # Project to detector
#       tmplt = nrbu.project_waveform(hp, hc, skyloc=(rec_right_ascension[s],
#           rec_declination[s]), polarization=rec_polarization[s],
#           detector_name=config.detector_name)
#
#       # Resize to the same length as the data
#       tlen = max(len(tmplt), len(reconstruction_data[s]))
#       tmplt.resize(tlen)
#
#       # Whiten the best-fit waveform
#       Tmplt = tmplt.to_frequencyseries()
#       Tmplt.data /= asd 
#
#       tmplt = Tmplt.to_timeseries()
#       tmplt.data /= pycbc.filter.sigma(tmplt)
#
#       sample = pycbc.types.TimeSeries(reconstruction_data[s],
#               delta_t=config.delta_t)
#       sample.data /= pycbc.filter.sigma(sample)
#
#       pl.figure()
#       pl.plot(tmplt.sample_times - tmplt.sample_times[np.argmax(tmplt)],
#               tmplt, label='tmplt')
#       pl.plot(sample.sample_times - sample.sample_times[np.argmax(sample)],
#               sample, label='data')
#       pl.legend()
#       pl.title('%f'%matches[w,s])
#       pl.xlim(-0.15,0.1)
#       pl.show()
#
#        sys.exit()
        # END XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


    bestidx=np.argmax(matches[w, :])

    print >> sys.stdout, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print >> sys.stdout, "Best Match:"

    chirp_mass = masses[w,bestidx]*simulations.simulations[w]['eta']**(3./5.)

    print >> sys.stdout, "Fit-factor: %.2f"%(matches[w,bestidx])
    print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
            chirp_mass, masses[w,bestidx], inclinations[w,bestidx])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dump data

filename=config.detector_name+'_'+opts.user_tag+'_'+config.algorithm+'_'+gpsnow+'.pickle'

# Dump results and configuration to pickle
pickle.dump([matches, masses, inclinations, config, simulations,
    __author__, __version__, __date__], open(filename, "wb"))



