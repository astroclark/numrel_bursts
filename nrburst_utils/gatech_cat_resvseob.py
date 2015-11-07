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
gatech_cat_resvseob.py

Compute matches between GATech NR waveforms at multiple resolutions and SEOBNRv2
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

import nrburst_utils as nrbu

from matplotlib import pyplot as pl

__author__ = "James Clark <james.clark@ligo.org>"

gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# Get the current git version
git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(sys.argv[0])).strip()
__version__ = "git id %s" % git_version_id

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse input

#
# --- catalog Definition
#
catdir='/home/jclark/Projects/ConvergenceTestData/CATALOG_PAPER_CONVERGENCE_TEST'
bounds = dict()

bounds['q'] = [2, 2]
#bounds['q'] = [4, 4]


#
# --- Plotting options
#
nMassPoints = 5
maxMass = 500.0

#
# --- Time Series Config
#
deltaT = 1./8192
datalen = 4.0

#
# --- Noise Spectrum
#
asd_file = \
        "/home/jclark/Projects/bhextractor/data/noise_curves/early_aligo.dat"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate The catalog

init_total_mass = 100   # Generate a catalog at this mass; shouldn't matter,
                        # we rescale anyway

distance=100. # Mpc (doesn't really matter)

plot_snr = 8

#
# --- Generate initial catalog
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()
simulations = \
        nrbu.simulation_details(param_bounds=bounds,
                catdir=catdir)

print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Building NR catalog'
print >> sys.stdout,  ''
catalog = nrbu.waveform_catalog(simulations, ref_mass=init_total_mass,
        SI_deltaT=deltaT, SI_datalen=datalen, distance=distance,
        trunc_time=False)
now = timeit.time.time()
print >> sys.stdout,  "...catalog construction took %.1f..."%(now-then)

asd_data = np.loadtxt(asd_file)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match Calculations
#

# For each waveform in the catalog:
#   1) GeIn [204]: nerate approximant(s) @ 5 mass scales between the min allowed mass / max
#      mass
#   2) Compute matches at the 5 mass scales
#   3) Append the 5 match values (per approximant) to the
#      simulations.simulations
#
#   By the end then, simulations.simulations is a list of dictionaries where
#   each dictionary is 1 GAtech waveform with all physical attributes, as well
#   as matches at 5 mass scales with some selection of approximants


# Loop over waves in NR catalog
for w, wave in enumerate(catalog.SIComplexTimeSeries):

    print >> sys.stdout,  "________________________________"
    print >> sys.stdout,  "Computing match for %s (%d/%d)"%(simulations.simulations[w]['wavename'],
            w+1, simulations.nsimulations)

    # Set up the Masses we're going to study
    masses = np.linspace(simulations.simulations[w]['Mmin30Hz'], maxMass,
            nMassPoints)

    # matches is going to be a list of tuples: (mass, match)
    matches = []

    # Get the NR (plus) wave and put it in a pycbc TimeSeries object
    hplus_NR = pycbc.types.TimeSeries(np.real(wave), delta_t=deltaT)
    hplus_NR.data[:] = nrbu.taper(hplus_NR.data[:], delta_t=hplus_NR.delta_t)


    # Extract physical parameters
    mass1, mass2 = nrbu.component_masses(init_total_mass, simulations.simulations[w]['q'])
    spin1z = simulations.simulations[w]['a1z']
    spin2z = simulations.simulations[w]['a2z']


    f, ax = pl.subplots(nrows = len(masses), ncols=2, figsize=(15,15))
    for m,mass in enumerate(masses):

        # --- Scale the NR waveform at this mass

        # Scale the NR waveform to the mass we want
        hplus_NR_new = pycbc.types.TimeSeries(nrbu.scale_wave(hplus_NR, mass,
            init_total_mass), delta_t=deltaT)
        hplus_NR_new.data[:] = nrbu.taper(hplus_NR_new.data[:],
                delta_t=hplus_NR_new.delta_t)

        # --- Generate the SEOBNR waveform to this mass

        mass1, mass2 = nrbu.component_masses(mass, simulations.simulations[w]['q'])

        # Estimate ffinal
        chi = spawaveform.computechi(mass1, mass2, spin1z, spin2z)
        ffinal = spawaveform.imrffinal(mass1, mass2, chi)

        Hf = hplus_NR_new.to_frequencyseries()
        f_lower = 0.8*Hf.sample_frequencies.data[ np.argmax(abs(Hf)) ]

        hplus_SEOBNR, _ = get_td_waveform(approximant="SEOBNRv2",
                distance=distance,
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z,
                f_lower=f_lower,
                delta_t=deltaT)

        hplus_SEOBNR.data = nrbu.taper(hplus_SEOBNR.data,
                delta_t=hplus_SEOBNR.delta_t)

     
        # Make the timeseries consistent lengths
        tlen = max(len(hplus_NR_new), len(hplus_SEOBNR))
        hplus_SEOBNR.resize(tlen)
        hplus_NR_new.resize(tlen)

        # Interpolate the ASD to the waveform frequencies (this is convenient so that we
        # end up with a PSD which overs all frequencies for use in the match calculation
        # later
        asd = np.interp(hplus_NR_new.to_frequencyseries().sample_frequencies,
                asd_data[:,0], asd_data[:,1])


        # Now insert ASD into a pycbc frequency series so we can use
        # pycbc.filter.match() later
        noise_psd = pycbc.types.FrequencySeries(asd**2, delta_f =
                hplus_NR_new.to_frequencyseries().delta_f)

#       from pycbc.psd import aLIGOZeroDetHighPower
#       delta_f = 1.0 / hplus_SEOBNR.duration
#       flen = tlen/2 + 1
#       noise_psd = aLIGOZeroDetHighPower(flen, delta_f, 10) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#       Hf = abs(hplus_SEOBNR.to_frequencyseries())
#       inband = noise_psd.sample_frequencies.data>30
#       upp_bound = \
#               noise_psd.sample_frequencies[inband][np.argwhere(Hf.data[inband]<1e-2*Hf[inband].max())[0]]
        #upp_bound = 0.5*1./deltaT
        upp_bound = 1.5*ffinal

        match, _ = pycbc.filter.match(hplus_SEOBNR, hplus_NR_new,
                low_frequency_cutoff=30.0, psd=noise_psd,
                high_frequency_cutoff=upp_bound)

        # ------------------------------------------------------------------
        # DIAGNOSTIC PLOTS

        print "~~~~~~~~~~~~~~~~~~~~~~~"
        print "Mass, mismatch (%)"
        print mass, 100*(1-match)

        # Normalise to unit SNR
        hplus_SEOBNR.data[:] /= pycbc.filter.sigma(hplus_SEOBNR,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        sigma_SEOBNR = pycbc.filter.sigma(hplus_SEOBNR,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        print 'sigma SEOBNR', sigma_SEOBNR

        hplus_NR_new.data[:] /= pycbc.filter.sigma(hplus_NR_new,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        sigma_NR_new = pycbc.filter.sigma(hplus_NR_new,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        print 'sigma NR', sigma_NR_new

        Hplus_SEOBNR = hplus_SEOBNR.to_frequencyseries()
        Hplus_NR_new = hplus_NR_new.to_frequencyseries()

        maxidx = np.argmax(hplus_NR_new)
        ax[m][0].plot(hplus_NR_new.sample_times -
                hplus_NR_new.sample_times[maxidx], hplus_NR_new,
                label='NR')
 
        maxidx = np.argmax(hplus_SEOBNR)
        ax[m][0].plot(hplus_SEOBNR.sample_times -
                hplus_SEOBNR.sample_times[maxidx], hplus_SEOBNR,
                label='SEOBNR')

        ax[m][0].set_title('M$_{\mathrm{tot}}$=%.2f M$_{\odot}$, mismatch=%.2f %%'%(
            mass, 100*(1-match)))

#        ax[m][0].legend(loc='lower left')
 
        ax[m][0].set_xlabel('Frequency [Hz]')
        ax[m][0].set_ylabel('h(t) [arb units]')


        ax[m][0].set_xlim(-2, 0.25)

        # Fdomain
 
        ax[m][1].loglog(Hplus_NR_new.sample_frequencies,
                   plot_snr*2*abs(Hplus_NR_new)*np.sqrt(Hplus_NR_new.sample_frequencies),
                   label='NR')
 
        ax[m][1].loglog(Hplus_SEOBNR.sample_frequencies,
                   plot_snr*2*abs(Hplus_SEOBNR)*np.sqrt(Hplus_SEOBNR.sample_frequencies),
                   label='SEOBNR')

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
 
    f.suptitle(simulations.simulations[w]['wavename'])
    f.tight_layout()
    f.savefig(simulations.simulations[w]['wavename']+'.png')

#    pl.show()
        # ------------------------------------------------------------------
#
#    sys.exit()






