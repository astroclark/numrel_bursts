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
burst_nr_match.py

Compute matches between burst reconstruction and NR waveforms
"""

import sys, os
from optparse import OptionParser
import ConfigParser
import subprocess
import cPickle as pickle

import numpy as np
import scipy.optimize
import timeit

import lal
from pylal import spawaveform
import pycbc.types
from pycbc.waveform import get_td_waveform
import pycbc.filter
from pycbc import pnutils


import nrburst_utils as nrbu

from matplotlib import pyplot as pl

__author__ = "James Clark <james.clark@ligo.org>"
gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# Get the current git version
#git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
#        cwd=os.path.dirname(sys.argv[0])).strip()
#__version__ = "git id %s" % git_version_id


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse input

#
# --- Catalogue Definition
#
bounds = dict()

# *************************************
#   1)  D12_q2.00_a0.15_-0.60_m200 
#    -  GATECH0199.h5
bounds['q'] = [1.999, 2.001]
bounds['spin1z'] = [0.14, 0.16]
bounds['spin2z'] = [-0.59, -0.61]
 
#    2) Sq4_d9_a0.6_oth.270_rr_M180 
#    - Highest resolution of this run missing in h5 catalog
#    - for lower res try: GATECH0410.h5
#   bounds['q'] = [3.9999, 4.0001]
#   bounds['spin1z'] = [0, 0.00001]
#   bounds['spin2z'] = [0, 0.00001]
#   bounds['spin1x'] = [-0.61, -0.59]
#   bounds['spin2x'] = [-0.61, -0.59]

#    3) D7.5_q15.00_a0.0_CHgEEB_m800
#    - missing from the h5 catalog
#bounds['q'] = [14.99, 15.01]

#    4) RO3_D10_q1.50_a0.60_oth.090_M120
#    - GATECH0173.h5
#bounds['q'] = [1.4999, 1.50001]
#bounds['spin1z'] = [0, 0.0001]
#bounds['spin2z'] = [0.59, 0.61]


# *************************************

inc = 0 # inclination
approx='SEOBNRv2'

#
#    4) RO3_D10_q1.50_a0.60_oth.090_M120
#    - GATECH0173.h5
#
#    5) q8_LL_D9_a0.6_th1_45_th2_225_m400
#    - GATECH0447.h5


#
# --- Plotting options
#
nMassPoints = 5
maxMass = 500.0

#
# --- Time Series Config
#
delta_t = 1./4096
datalen = 4.0

#
# --- Noise Spectrum
#
asd_file = \
        "/home/jclark/Projects/bhextractor/data/noise_curves/early_aligo.dat"

#
# --- Catalog
#
catalog='/home/jclark/Projects/GW150914_data/nr_catalog/gatech_hdf5'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate The Catalogue


plot_snr = 8

#
# --- Generate initial catalogue
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()
simulations = \
        nrbu.simulation_details(param_bounds=bounds,
                catdir=catalog)

asd_data = np.loadtxt(asd_file)

sys.exit()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match Calculations
#

# For each waveform in the catalogue:
#   1) GeIn [204]: nerate approximant(s) @ 5 mass scales between the min allowed mass / max
#      mass
#   2) Compute matches at the 5 mass scales
#   3) Append the 5 match values (per approximant) to the
#      simulations.simulations
#
#   By the end then, simulations.simulations is a list of dictionaries where
#   each dictionary is 1 GAtech waveform with all physical attributes, as well
#   as matches at 5 mass scales with some selection of approximants

# Set up the Masses we're going to study
masses = np.linspace(simulations.simulations[0]['Mmin30Hz'], maxMass,
        nMassPoints)

# matches is going to be a list of tuples: (mass, match)
matches = []

f, ax = pl.subplots(nrows = len(masses), ncols=2, figsize=(15,15))

for m,mass in enumerate(masses):
    "Extracting and generating mass %d of %d (%.2f)"%(m, len(masses), mass)

    # --- Generate the polarisations
    hplus_NR, hcross_NR = \
            nrbu.get_wf_pols(simulations.simulations[0]['wavefile'], mass,
                    inclination=inc, delta_t=delta_t, f_lower=30.00001)

    # --- Generate the approx waveform to this mass
    mass1, mass2 = nrbu.component_masses(mass, simulations.simulations[0]['q'])

    # Estimate ffinal 
    chi = pnutils.phenomb_chi(mass1, mass2,
            simulations.simulations[0]['spin1z'],simulations.simulations[0]['spin2z'])
    ffinal = pnutils.get_final_freq(approx, mass1, mass2, 
            simulations.simulations[0]['spin1z'],simulations.simulations[0]['spin2z'])

    Hf = hplus_NR.to_frequencyseries()
    f_lower = 0.8*Hf.sample_frequencies.data[ np.argmax(abs(Hf)) ]

    hplus_approx, _ = get_td_waveform(approximant=approx,
            distance=100,
            mass1=mass1,
            mass2=mass2,
            spin1z=simulations.simulations[0]['spin1z'],
            spin2z=simulations.simulations[0]['spin2z'],
            f_lower=f_lower,
            delta_t=delta_t)

    hplus_approx.data = nrbu.taper(hplus_approx.data,
            delta_t=hplus_approx.delta_t)

 
    # Make the timeseries consistent lengths
    tlen = max(len(hplus_NR), len(hplus_approx))
    hplus_approx.resize(tlen)
    hplus_NR.resize(tlen)

    # Interpolate the ASD to the waveform frequencies (this is convenient so that we
    # end up with a PSD which overs all frequencies for use in the match calculation
    # later
    asd = np.interp(hplus_NR.to_frequencyseries().sample_frequencies,
            asd_data[:,0], asd_data[:,1])


    # Now insert ASD into a pycbc frequency series so we can use
    # pycbc.filter.match() later
    noise_psd = pycbc.types.FrequencySeries(asd**2, delta_f =
            hplus_NR.to_frequencyseries().delta_f)

#       from pycbc.psd import aLIGOZeroDetHighPower
#       delta_f = 1.0 / hplus_approx.duration
#       flen = tlen/2 + 1
#       noise_psd = aLIGOZeroDetHighPower(flen, delta_f, 10) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#       Hf = abs(hplus_approx.to_frequencyseries())
#       inband = noise_psd.sample_frequencies.data>30
#       upp_bound = \
#               noise_psd.sample_frequencies[inband][np.argwhere(Hf.data[inband]<1e-2*Hf[inband].max())[0]]
    #upp_bound = 0.5*1./delta_t
    upp_bound = 1.5*ffinal

    match, _ = pycbc.filter.match(hplus_approx, hplus_NR,
            low_frequency_cutoff=30.0, psd=noise_psd,
            high_frequency_cutoff=upp_bound)

    # ------------------------------------------------------------------
    # DIAGNOSTIC PLOTS

    print "~~~~~~~~~~~~~~~~~~~~~~~"
    print "Mass, mismatch (%)"
    print mass, 100*(1-match)

    # Normalise to unit SNR
    hplus_approx.data[:] /= pycbc.filter.sigma(hplus_approx,
            psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
    sigma_approx = pycbc.filter.sigma(hplus_approx,
            psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
    print 'sigma approx', sigma_approx

    hplus_NR.data[:] /= pycbc.filter.sigma(hplus_NR,
            psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
    sigma_NR = pycbc.filter.sigma(hplus_NR,
            psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
    print 'sigma NR', sigma_NR

    Hplus_approx = hplus_approx.to_frequencyseries()
    Hplus_NR = hplus_NR.to_frequencyseries()

    maxidx = np.argmax(hplus_NR)
    ax[m][0].plot(hplus_NR.sample_times -
            hplus_NR.sample_times[maxidx], hplus_NR,
            label='NR')

    maxidx = np.argmax(hplus_approx)
    ax[m][0].plot(hplus_approx.sample_times -
            hplus_approx.sample_times[maxidx], hplus_approx,
            label='approx')

    ax[m][0].set_title('M$_{\mathrm{tot}}$=%.2f M$_{\odot}$, mismatch=%.2f %%'%(
        mass, 100*(1-match)))

#        ax[m][0].legend(loc='lower left')

    ax[m][0].set_xlabel('Frequency [Hz]')
    ax[m][0].set_ylabel('h(t) [arb units]')


    ax[m][0].set_xlim(-2, 0.25)

    # Fdomain

    ax[m][1].loglog(Hplus_NR.sample_frequencies,
               plot_snr*2*abs(Hplus_NR)*np.sqrt(Hplus_NR.sample_frequencies),
               label='NR')

    ax[m][1].loglog(Hplus_approx.sample_frequencies,
               plot_snr*2*abs(Hplus_approx)*np.sqrt(Hplus_approx.sample_frequencies),
               label=approx)

    ax[m][1].loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd),
            label='noise psd', color='k', linestyle='--')

    ax[m][1].set_title('M$_{\mathrm{tot}}$=%.2f M$_{\odot}$, mismatch=%.2f %%'%(
        mass, 100*(1-match)))

    ax[m][1].legend(loc='lower right')

    ax[m][1].axvline(30, color='r')
    ax[m][1].axvline(upp_bound, color='r')

    ax[m][1].set_xlabel('Frequency [Hz]')
    ax[m][1].set_ylabel('2|H$_+$($f$)|$\sqrt{f}$ & $\sqrt{S(f)}$')
    ax[m][1].set_ylim(0.01*min(asd), 10*max(asd))
    ax[m][1].set_xlim(9, 2e3)

f.tight_layout()
pl.show()
    # ------------------------------------------------------------------
#
sys.exit()






