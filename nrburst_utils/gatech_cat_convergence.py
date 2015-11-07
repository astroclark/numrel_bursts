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
gatech_cat_convergence.py

Compute matches between GATech waveforms at multiple resolutions
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

# Pick first (lowest resolution) waveform then loop through remaining waveforms
# computing matches at different mass scales.  E.g.,:
#
#   1st iteration:  (m120|m140) @ 50, 100, 150 Msun
#   2nd iteration:  (m120|m160) @ 50, 100, 150 Msun
#   3rd iteration:  (m120|m180) @ 50, 100, 150 Msun



# Get the NR (plus) wave and put it in a pycbc TimeSeries object
min_res = simulations.simulations[0]['mres']
hplus_NRa = pycbc.types.TimeSeries(np.real(catalog.SIComplexTimeSeries[0,:]),
        delta_t=deltaT)
hplus_NRa.data[:] = nrbu.taper(hplus_NRa.data[:], delta_t=hplus_NRa.delta_t)

match_data = []

# Loop over waves in NR catalog
for w in xrange(1,simulations.nsimulations):

    print >> sys.stdout,  "________________________________"
    print >> sys.stdout,  "Computing match for (%s|%s) [%s] (%d/%d)"%(
            min_res, simulations.simulations[w]['mres'],
            simulations.simulations[w]['wavename'],
            w, simulations.nsimulations-1)

    # Set up the Masses we're going to study
    masses = np.linspace(simulations.simulations[w]['Mmin30Hz'], maxMass,
            nMassPoints)

    # matches is going to be a 2D array: (mass, match)
    matches = np.zeros(shape=(len(masses), 2))

    # Extract this resolution from the catalog
    hplus_NRb = pycbc.types.TimeSeries(np.real(catalog.SIComplexTimeSeries[w,:]),
            delta_t=deltaT)
    hplus_NRb.data[:] = nrbu.taper(hplus_NRb.data[:], delta_t=hplus_NRb.delta_t)

    # Extract physical parameters (useful for data conditioning)
    mass1, mass2 = nrbu.component_masses(init_total_mass, simulations.simulations[w]['q'])
    spin1z = simulations.simulations[w]['a1z']
    spin2z = simulations.simulations[w]['a2z']

    f, ax = pl.subplots(nrows = len(masses), ncols=2, figsize=(15,15))
    for m,mass in enumerate(masses):
        print "... At mass %f "%mass

        # --- Scale the NR waveform at this mass

        # Scale the NR waveform to the mass we want
        hplus_NRa_scaled = pycbc.types.TimeSeries(nrbu.scale_wave(hplus_NRa, mass,
            init_total_mass), delta_t=deltaT)
        hplus_NRa_scaled.data[:] = nrbu.taper(hplus_NRa_scaled.data[:],
                delta_t=hplus_NRa_scaled.delta_t)

        hplus_NRb_scaled = pycbc.types.TimeSeries(nrbu.scale_wave(hplus_NRb, mass,
            init_total_mass), delta_t=deltaT)
        hplus_NRb_scaled.data[:] = nrbu.taper(hplus_NRb_scaled.data[:],
                delta_t=hplus_NRb_scaled.delta_t)

        # Estimate ffinal
        mass1, mass2 = nrbu.component_masses(mass, simulations.simulations[w]['q'])
        chi = spawaveform.computechi(mass1, mass2, spin1z, spin2z)
        ffinal = spawaveform.imrffinal(mass1, mass2, chi)
        upp_bound = 1.5*ffinal

        Hf = hplus_NRa_scaled.to_frequencyseries()
        f_lower = 0.8*Hf.sample_frequencies.data[ np.argmax(abs(Hf)) ]


        # Interpolate the ASD to the waveform frequencies (this is convenient so that we
        # end up with a PSD which overs all frequencies for use in the match calculation
        # later
        asd = np.interp(hplus_NRa_scaled.to_frequencyseries().sample_frequencies,
                asd_data[:,0], asd_data[:,1])


        # Now insert ASD into a pycbc frequency series so we can use
        # pycbc.filter.match() later
        noise_psd = pycbc.types.FrequencySeries(asd**2, delta_f =
                hplus_NRa_scaled.to_frequencyseries().delta_f)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        match, _ = pycbc.filter.match(hplus_NRa_scaled, hplus_NRb_scaled,
                low_frequency_cutoff=30.0, psd=noise_psd,
                high_frequency_cutoff=upp_bound)

        matches[m,0] = mass
        matches[m,1] = match

        # ------------------------------------------------------------------
        # DIAGNOSTIC PLOTS

        print "~~~~~~~~~~~~~~~~~~~~~~~"
        print "Mass, mismatch (%)"
        print mass, 100*(1-match)

        # Normalise to unit SNR
        hplus_NRa_scaled.data[:] /= pycbc.filter.sigma(hplus_NRa_scaled,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        sigma_NRa_scaled = pycbc.filter.sigma(hplus_NRa_scaled,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        print 'sigma NRa', sigma_NRa_scaled

        # Normalise to unit SNR
        hplus_NRb_scaled.data[:] /= pycbc.filter.sigma(hplus_NRb_scaled,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        sigma_NRb_scaled = pycbc.filter.sigma(hplus_NRb_scaled,
                psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
        print 'sigma NRb', sigma_NRb_scaled

        Hplus_NRa_scaled = hplus_NRa_scaled.to_frequencyseries()
        Hplus_NRb_scaled = hplus_NRb_scaled.to_frequencyseries()

        # --- Tdomain

        maxidx = np.argmax(hplus_NRa_scaled)
        ax[m][0].plot(hplus_NRa_scaled.sample_times -
                hplus_NRa_scaled.sample_times[maxidx], hplus_NRa_scaled,
                label='$\delta$m=%d'%min_res)

        maxidx = np.argmax(hplus_NRb_scaled)
        ax[m][0].plot(hplus_NRb_scaled.sample_times -
                hplus_NRb_scaled.sample_times[maxidx], hplus_NRb_scaled,
                label='$\delta$m=%d'%simulations.simulations[w]['mres'])

        ax[m][0].set_title('M$_{\mathrm{tot}}$=%.2f M$_{\odot}$, mismatch=%.2f %%'%(
            mass, 100*(1-match)))
 
        ax[m][0].set_xlabel('Frequency [Hz]')
        ax[m][0].set_ylabel('h(t) [arb units]')

        ax[m][0].set_xlim(-2, 0.25)

        # --- Fdomain
 
        ax[m][1].loglog(Hplus_NRa_scaled.sample_frequencies,
                   plot_snr*2*abs(Hplus_NRa_scaled)*np.sqrt(Hplus_NRa_scaled.sample_frequencies),
                   label='$\delta$m=%d'%min_res)

        ax[m][1].loglog(Hplus_NRb_scaled.sample_frequencies,
                   plot_snr*2*abs(Hplus_NRb_scaled)*np.sqrt(Hplus_NRb_scaled.sample_frequencies),
                   label='$\delta$m=%d'%simulations.simulations[w]['mres'])

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
    f.savefig('converge_'+simulations.simulations[w]['wavename']+'.png')
    pl.close()

    match_data.append((simulations.simulations[w]['mres'], matches))


# Now plot it all
f, ax = pl.subplots()
for m, (mres, matches) in enumerate(match_data):
    ax.plot(matches[:,0], 1-matches[:,1], '-o',
            label='(%d|%d)'%(min_res, simulations.simulations[m+1]['mres']))

ax.legend()
ax.set_xlabel('Total Mass [M$_{\odot}$]')
ax.set_ylabel('1-match')
ax.legend()
ax.minorticks_on()
ax.set_title('%s'%simulations.simulations[0]['wavename'])

f.savefig('converge_mismatchvsmass_%s.png'%simulations.simulations[0]['wavename'])

pl.show()




