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
#gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = 'today'#subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# Get the current git version
git_version_id = 'this one'#subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        #cwd=os.path.dirname(sys.argv[0])).strip()
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
h1_reconstruction_data = np.loadtxt(config.h1_reconstruction)
h1_asd_data = np.loadtxt(config.h1_spectral_estimate)
l1_reconstruction_data = np.loadtxt(config.l1_reconstruction)
l1_asd_data = np.loadtxt(config.l1_spectral_estimate)

rec_ext_params = np.loadtxt(config.extrinsic_params)

# If BayesWave, select the user-specified number of samples for which we will
# compute matches (useful for speed / development work)
if config.algorithm=='BW':

    rec_right_ascension = rec_ext_params[:,2] / lal.PI_180
    rec_declination     = np.arcsin(rec_ext_params[:,3]) / lal.PI_180
    rec_polarization    = rec_ext_params[:,4] / lal.PI_180

    if config.nsampls != 'all':

        # Load sampled waveforms
        print 'reducing sample size'
        idx = np.random.random_integers(low=0, high=len(h1_reconstruction_data)-1,
                size=config.nsampls)

        h1_reconstruction_data = h1_reconstruction_data[idx]
        l1_reconstruction_data = l1_reconstruction_data[idx]
        rec_right_ascension = rec_right_ascension[idx]
        rec_declination     = rec_declination[idx]
        rec_polarization    = rec_polarization[idx]
    else:
        print 'using ALL BW samples (%d)'%len(h1_reconstruction_data)
        setattr(config, 'nsampls', len(h1_reconstruction_data))


elif config.algorithm=='CWB':

    sky_loc_geographic = lal.SkyPosition()
    sky_loc_geographic.latitude = rec_ext_params[0]
    sky_loc_geographic.longitude = rec_ext_params[1]
    sky_loc_geographic.system=lal.COORDINATESYSTEM_GEOGRAPHIC

    sky_loc_equatorial = lal.SkyPosition()
    sky_loc_equatorial.system = lal.COORDINATESYSTEM_EQUATORIAL
    lal.GeographicToEquatorial(sky_loc_equatorial, sky_loc_geographic,
            lal.LIGOTimeGPS(1126259462))


    rec_right_ascension = [sky_loc_equatorial.longitude]
    rec_declination = [sky_loc_equatorial.latitude]
    rec_polarization = [rec_ext_params[2]]

    h1_reconstruction_data = [nrbu.extract_wave(h1_reconstruction_data,
        config.datalen, config.sample_rate)]

    l1_reconstruction_data = [nrbu.extract_wave(l1_reconstruction_data,
        config.datalen, config.sample_rate)]

    setattr(config, 'nsampls', 1)

elif config.algorithm=='HWINJ':

    rec_right_ascension = [rec_ext_params[0] / lal.PI_180]
    rec_declination     = [rec_ext_params[1] / lal.PI_180]
    rec_polarization    = [rec_ext_params[2] / lal.PI_180]

    h1_reconstruction_data = [nrbu.extract_wave(reconstruction_data,
        config.datalen, config.sample_rate)]
    l1_reconstruction_data = [nrbu.extract_wave(reconstruction_data,
        config.datalen, config.sample_rate)]

    setattr(config, 'nsampls', 1)

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

if getattr(opts, 'simulation_number') != "all":
    setattr(simulations, 'simulations',
            [simulations.simulations[opts.simulation_number]])
    setattr(simulations, 'nsimulations', len(simulations.simulations))

# Useful time/freq samples
time_axis = np.arange(config.datalen, config.delta_t)
freq_axis = np.arange(0.5*config.datalen/config.delta_t+1./config.datalen) * 1./config.datalen

# Interpolate the ASD to the waveform frequencies (this is convenient so that we
# end up with a PSD which overs all frequencies for use in the match calculation
# later - In practice, this will really just pad out the spectrum at low
# frequencies)
h1_asd = np.interp(freq_axis, h1_asd_data[:,0], h1_asd_data[:,1])
l1_asd = np.interp(freq_axis, l1_asd_data[:,0], l1_asd_data[:,1])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameter Estimation
#

#
# --- Compute Fitting-Factor for each NR waveform
#
# Fitting factor: normalised inner product, maximised over time, phase-offset,
# total mass and orientation

# Preallocate
matches = np.zeros(shape=(3, simulations.nsimulations, config.nsampls))
masses  = np.zeros(shape=(3, simulations.nsimulations, config.nsampls))
inclinations  = np.zeros(shape=(3, simulations.nsimulations, config.nsampls))


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


    for s, (h1_sampled_waveform, l1_sampled_waveform) in \
            enumerate(zip(h1_reconstruction_data, l1_reconstruction_data)):


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
#       h1_data, _ = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'], mass,
#               inclination=inc, delta_t=config.delta_t)
#       l1_data, _ = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'], mass,
#               inclination=inc, delta_t=config.delta_t)
#
#       # Project to detector
#       #h1_data = nrbu.project_waveform(hp, hc, skyloc=None,
#               #polarization=None, detector_name="H1")
#
#       # Resize to the same length as the data
#       h1_data.resize(config.datalen*config.sample_rate)
#       l1_data.resize(config.datalen*config.sample_rate)
#
#       # Whiten the template
#       h1_Data = h1_data.to_frequencyseries()
#       h1_Data.data /= h1_asd 
#       h1_sampled_waveform = h1_Data.to_timeseries().data[:]
#
#       l1_Data = l1_data.to_frequencyseries()
#       l1_Data.data /= l1_asd 
#       l1_sampled_waveform = l1_Data.to_timeseries().data[:]
#
#       print "INJECTING:"
#       print mass, inc, rec_polarization[s]
#

        # END XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


        print >> sys.stdout, '-----------------------------'
        print >> sys.stdout, "Evaluating sample waveform %d of %d"%( s,
                len(h1_reconstruction_data) )
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

        # ################### H1 ################ #
        print "--- Analysing H1 ---"
        h1_result = scipy.optimize.fmin(nrbu.single_ifo_mismatch,
                x0=init_guess,
                args=(
                    simulations.simulations[w]['wavefile'],
                    None, None,
                    'H1',
                    (min_mass, max_mass),
                    h1_sampled_waveform, h1_asd, config.delta_t
                    ), xtol=1e-3, ftol=1e-3, maxfun=10000,
                full_output=True, retall=True, disp=True)

        now = timeit.time.time()
        print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

        matches[1,w,s] = 1-h1_result[1]
        masses[1,w,s]  = h1_result[0][0]
        inclinations[1,w,s]  = h1_result[0][1]

        chirp_mass = masses[1,w,s]*simulations.simulations[w]['eta']**(3./5.)

        print >> sys.stdout, ""
        print >> sys.stdout, "Fit-factor: %.2f"%(matches[1,w,s])
        print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
                chirp_mass, masses[1,w,s], inclinations[1,w, s])
        print >> sys.stdout, ""

        # ################### L1 ################ #
        print "--- Analysing L1 ---"
        l1_result = scipy.optimize.fmin(nrbu.single_ifo_mismatch,
                x0=init_guess,
                args=(
                    simulations.simulations[w]['wavefile'],
                    None, None,
                    'L1',
                    (min_mass, max_mass),
                    l1_sampled_waveform, l1_asd, config.delta_t
                    ), xtol=1e-3, ftol=1e-3, maxfun=10000,
                full_output=True, retall=True, disp=True)

        now = timeit.time.time()
        print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

        matches[2,w,s] = 1-l1_result[1]
        masses[2,w,s]  = l1_result[0][0]
        inclinations[2,w,s]  = l1_result[0][1]

        chirp_mass = masses[2,w,s]*simulations.simulations[w]['eta']**(3./5.)

        print >> sys.stdout, ""
        print >> sys.stdout, "Fit-factor: %.2f"%(matches[2,w,s])
        print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
                chirp_mass, masses[2,w,s], inclinations[2,w,s])
        print >> sys.stdout, ""


        # ################### HL ################ #

        print "--- Analysing HL Network ---"

        hl_result = scipy.optimize.fmin(nrbu.network_mismatch,
                x0=init_guess,
                args=(
                    simulations.simulations[w]['wavefile'],
                    None, None,
                    (min_mass, max_mass),
                    h1_sampled_waveform, h1_asd,
                    l1_sampled_waveform, l1_asd, config.delta_t
                    ), xtol=1e-3, ftol=1e-3, maxfun=10000,
                full_output=True, retall=True, disp=True)

        now = timeit.time.time()
        print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

        matches[0,w,s] = 1-hl_result[1]
        masses[0,w,s]  = l1_result[0][0]
        inclinations[0,w,s]  = hl_result[0][1]

        chirp_mass = masses[0,w,s]*simulations.simulations[w]['eta']**(3./5.)

        print >> sys.stdout, ""
        print >> sys.stdout, "Fit-factor: %.2f"%(matches[0,w,s])
        print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
                chirp_mass, masses[0,w,s], inclinations[0,w,s])
        print >> sys.stdout, ""

        # START XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XXX: Debugging / Testing

#       from matplotlib import pyplot as pl
#       import pycbc.types
#
#       # Generate the polarisations which correspond to the best-fit parameters
#       h1_tmplt, _ = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'],
#               masses[1,w,s], inclination=inclinations[1,w,s],
#               delta_t=config.delta_t)
#       l1_tmplt, _ = nrbu.get_wf_pols(simulations.simulations[w]['wavefile'],
#               masses[2,w,s], inclination=inclinations[2,w,s],
#               delta_t=config.delta_t)
#
#       # Project to detector
#       #h1_tmplt = nrbu.project_waveform(hp, hc, skyloc=None, polarization=None,
#       #        detector_name="H1")
#
#       # Resize to the same length as the data
#       tlen = max(len(h1_tmplt), len(h1_reconstruction_data[s]))
#       h1_tmplt.resize(tlen)
#
#       # Whiten the best-fit waveform
#       h1_Tmplt = h1_tmplt.to_frequencyseries()
#       h1_Tmplt.data /= h1_asd 
#
#       h1_tmplt = h1_Tmplt.to_timeseries()
#       h1_tmplt.data /= pycbc.filter.sigma(h1_tmplt)
#
#       h1_sample = pycbc.types.TimeSeries(h1_sampled_waveform,
#               delta_t=config.delta_t)
#       h1_sample.data /= pycbc.filter.sigma(h1_sample)
#
#       pl.figure()
#       pl.plot(h1_tmplt.sample_times - h1_tmplt.sample_times[np.argmax(h1_tmplt)],
#               h1_tmplt, label='H1 tmplt')
#       pl.plot(h1_sample.sample_times - h1_sample.sample_times[np.argmax(h1_sample)],
#               h1_sample, label='H1 data')
#       pl.legend()
#       pl.title('%f'%matches[1,w,s])
#       pl.xlim(-0.15,0.1)
#       pl.show()
#
#       # Resize to the same length as the data
#       tlen = max(len(l1_tmplt), len(l1_reconstruction_data[s]))
#       l1_tmplt.resize(tlen)
#
#       # Whiten the best-fit waveform
#       l1_Tmplt = l1_tmplt.to_frequencyseries()
#       l1_Tmplt.data /= l1_asd 
#
#       l1_tmplt = l1_Tmplt.to_timeseries()
#       l1_tmplt.data /= pycbc.filter.sigma(l1_tmplt)
#
#       l1_sample = pycbc.types.TimeSeries(l1_sampled_waveform,
#               delta_t=config.delta_t)
#       l1_sample.data /= pycbc.filter.sigma(l1_sample)
#
#       pl.figure()
#       pl.plot(l1_tmplt.sample_times - l1_tmplt.sample_times[np.argmax(l1_tmplt)],
#               l1_tmplt, label='L1 tmplt')
#       pl.plot(l1_sample.sample_times - l1_sample.sample_times[np.argmax(l1_sample)],
#               l1_sample, label='L1 data')
#       pl.legend()
#       pl.title('%f'%matches[1,w,s])
#       pl.xlim(-0.15,0.1)
#       pl.show()
#
#       sys.exit()
        # END XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


    hl_bestidx=np.argmax(matches[0, w, :])
    h_bestidx=np.argmax(matches[1, w, :])
    l_bestidx=np.argmax(matches[2, w, :])

    print >> sys.stdout, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print >> sys.stdout, "HL Best Match:"

    chirp_mass = masses[w,bestidx]*simulations.simulations[w]['eta']**(3./5.)

    print >> sys.stdout, "Fit-factor: %.2f"%(matches[0,w,hl_bestidx])
    print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
            chirp_mass, masses[0,w,bestidx], inclinations[0,w,bestidx])

    print >> sys.stdout, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print >> sys.stdout, "H1 Best Match:"

    chirp_mass = masses[1,w,bestidx]*simulations.simulations[w]['eta']**(3./5.)

    print >> sys.stdout, "Fit-factor: %.2f"%(matches[1,w,bestidx])
    print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
            chirp_mass, masses[1,w,bestidx], inclinations[1,w,bestidx])

    print >> sys.stdout, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print >> sys.stdout, "L1 Best Match:"

    chirp_mass = masses[2,w,bestidx]*simulations.simulations[w]['eta']**(3./5.)

    print >> sys.stdout, "Fit-factor: %.2f"%(matches[2,w,bestidx])
    print >> sys.stdout, "Mchirp=%.2f,  Mtot=%.2f, inclination=%.2f"%(
            chirp_mass, masses[2,w,bestidx], inclinations[2,w,bestidx])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dump data

filename=config.detector_name+'_'+opts.user_tag+'_'+config.algorithm+'_nrsim-'+str(opts.simulation_number)+'.pickle'

# Dump results and configuration to pickle
pickle.dump([matches, masses, inclinations, config, simulations,
    __author__, __version__, __date__], open(filename, "wb"))



