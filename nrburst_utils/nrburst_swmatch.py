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

    if config.nsampls != 'all':

        # Load sampled waveforms
        print 'reducing sample size'
        idx = np.random.random_integers(low=0,
                high=len(h1_reconstruction_data)-1,
                size=config.nsampls)

        h1_reconstruction_data = h1_reconstruction_data[idx]
        l1_reconstruction_data = l1_reconstruction_data[idx]

    elif opts.max_sample is not None:

        print "selecting out samples %d:%d"%(opts.min_sample, opts.max_sample)
        idx = range(opts.min_sample, opts.max_sample+1)

        h1_reconstruction_data = h1_reconstruction_data[idx]
        l1_reconstruction_data = l1_reconstruction_data[idx]

    else:
        print 'using ALL BW samples (%d)'%len(h1_reconstruction_data)

    setattr(config, 'nsampls', len(h1_reconstruction_data))


elif config.algorithm=='CWB':

    h1_reconstruction_data = [nrbu.extract_wave(h1_reconstruction_data,
        config.datalen, config.sample_rate)]

    l1_reconstruction_data = [nrbu.extract_wave(l1_reconstruction_data,
        config.datalen, config.sample_rate)]

    setattr(config, 'nsampls', 1)


# Useful time/freq samples
time_axis = np.arange(config.datalen, config.delta_t)
freq_axis = np.arange(0.5*config.datalen/config.delta_t+1./config.datalen) * 1./config.datalen

# Interpolate the ASD to the waveform frequencies (this is convenient so that we
# end up with a PSD which overs all frequencies for use in the match calculation
# later - In practice, this will really just pad out the spectrum at low
# frequencies)
h1_asd = np.exp(np.interp(np.log(freq_axis), np.log(h1_asd_data[:,0]),
    np.log(h1_asd_data[:,1])))
l1_asd = np.exp(np.interp(np.log(freq_axis), np.log(l1_asd_data[:,0]),
    np.log(l1_asd_data[:,1])))


# Load the Software Injection
h1_sw_injection=np.loadtxt(cp.get('paths', 'h1_injection'))
l1_sw_injection=np.loadtxt(cp.get('paths', 'l1_injection'))

# Reduce to specified datalen
h1_sw_injection = nrbu.extract_wave(h1_sw_injection, config.datalen,
        config.sample_rate)
l1_sw_injection = nrbu.extract_wave(l1_sw_injection, config.datalen,
        config.sample_rate)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameter Estimation
#


# Preallocate
matches = np.zeros(config.nsampls)

for s, (h1_sampled_waveform, l1_sampled_waveform) in \
        enumerate(zip(h1_reconstruction_data, l1_reconstruction_data)):


    print >> sys.stdout, '-----------------------------'
    print >> sys.stdout, "Evaluating sample waveform %d of %d"%(s,
            len(h1_reconstruction_data) )

    then = timeit.time.time()

    # ################### HL ################ #

    print "--- Analysing HL Network ---"

    now = timeit.time.time()
    print >> sys.stdout,  "...mass optimisation took %.3f sec..."%(now-then)

    matches[s] = nrbu.network_sw_match(h1_sw_injection, l1_sw_injection,
            h1_sampled_waveform, l1_sampled_waveform, delta_t=config.delta_t,
            f_min=config.f_min)

    print >> sys.stdout, ""
    print >> sys.stdout, "Fit-factor: %.2f"%(matches[s])
    print >> sys.stdout, ""


hl_bestidx=np.argmax(matches)

print >> sys.stdout, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print >> sys.stdout, "HL Best Match:"

print >> sys.stdout, "Fit-factor: %.2f"%(matches[hl_bestidx])




