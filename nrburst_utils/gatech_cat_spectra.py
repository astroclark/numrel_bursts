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
from scipy import signal
import timeit

import lal
from pylal import spawaveform
import pycbc.types
from pycbc.waveform import get_td_waveform
from pycbc.waveform import utils as wfutils
import pycbc.filter
from pycbc import pnutils


import nrburst_utils as nrbu

from matplotlib import pyplot as pl

__author__ = "James Clark <james.clark@ligo.org>"
gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

def window_wave(input_data):

    nonzero=np.argwhere(abs(input_data)>1e-3*max(abs(input_data)))
    idx = range(nonzero[0],nonzero[-1])
    win = planckwin(len(idx), 0.3)
    win[0.5*len(win):] = 1.0
    input_data[idx] *= win

    return input_data

def planckwin(N, epsilon):

    t1 = -0.5*N
    t2 = -0.5*N * (1.-2.*epsilon)
    t3 = 0.5*N * (1.-2.*epsilon)
    t4 = 0.5*N

    Zp = lambda t: (t2-t1)/(t-t1) + (t2-t1)/(t-t2)
    Zm = lambda t: (t3-t4)/(t-t3) + (t3-t4)/(t-t4)

    win = np.zeros(N)
    ts = np.arange(-0.5*N, 0.5*N)

    for n,t in enumerate(ts):
        if t<=t1:
            win[n] = 0.0
        elif t1<t<t2:
            win[n] = 1./(np.exp(Zp(t))+1)
        elif t2<=t<=t3:
            win[n] = 1.0
        elif t3<t<t4:
            win[n] = 1./(np.exp(Zm(t))+1)

    return win


def scale_NR(times_codeunits, wave, mass, delta_t=1./2048):
    """
    Scale the waveform to total_mass.  Assumes the waveform is initially
    generated at init_total_mass defined in this script.
    """

    peakidx = np.argmax(abs(wave))

    NR_deltaT = np.diff(times_codeunits)[0]
    NR_datalen = len(times_codeunits)
    SI_deltaT_of_NR = mass * lal.MTSUN_SI * NR_deltaT

    old_times = np.arange(0, NR_datalen*SI_deltaT_of_NR, SI_deltaT_of_NR)
    interp_times = np.arange(0, NR_datalen*SI_deltaT_of_NR, delta_t)

    resampled_wave = np.interp(interp_times, old_times, wave)

    return pycbc.types.TimeSeries(resampled_wave, delta_t)

def parse_NR_asc(file):
    data = np.loadtxt(file)
    times_codeunits     = np.loadtxt(file)[:,0]
    hplus_NR_codeunits  = data[:,1]
    hcross_NR_codeunits = data[:,2]
    dAmpbyAmp_codeunits = data[:,3]
    dphi_codeunits      = data[:,4]

    return (times_codeunits, hplus_NR_codeunits, hcross_NR_codeunits,
            dAmpbyAmp_codeunits, dphi_codeunits)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse input

sim_number = int(sys.argv[1])

#
# --- Catalogue Definition
#
bounds = None
#bounds = dict()
#bounds['Mchirpmin30Hz'] = [-np.inf, 27]
#bounds['a1'] = [-0.01, 0.01]
#bounds['a2'] = [-0.01, 0.01]
#bounds['q'] = [0.99, 1.01]


#
# --- Setup the approximant we want
#
inc = 0 # inclination
approx='SEOBNRv2'
f_low_approx=10


#
# --- Time Series Config
#
delta_t = 1./4096
datalen = 16

#
# --- Noise Spectrum
#
asd_file = \
        "/home/jclark/GW150914_data/noise_curves/early_aligo.dat"

#
# --- Catalog
#
catalog='/home/jclark/GW150914_data/nr_catalog/gatech_hdf5'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate The Catalogue


distance = 500
plot_snr = 8

#
# --- Generate initial catalogue
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()
simulations = \
        nrbu.simulation_details(param_bounds=bounds, catdir=catalog)
filename = simulations.simulations[sim_number]['wavefile'].split('/')[-1]

asd_data = np.loadtxt(asd_file)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match Calculation & plotting
#

print "Extracting and generating waveform"

mass = simulations.simulations[sim_number]['Mmin30Hz']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Common params

mass1, mass2 = pnutils.mtotal_eta_to_mass1_mass2(mass,
        simulations.simulations[sim_number]['eta'])

# Estimate ffinal 
chi = pnutils.phenomb_chi(mass1, mass2,
        simulations.simulations[sim_number]['spin1z'],simulations.simulations[sim_number]['spin2z'])
ffinal = pnutils.get_final_freq(approx, mass1, mass2, 
        simulations.simulations[sim_number]['spin1z'],simulations.simulations[sim_number]['spin2z'])
#upp_bound = ffinal
#upp_bound = 1.5*ffinal
upp_bound = 0.5/delta_t

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NUMERICAL RELATIVITY

# --- Generate the polarisations
hplus_NR, hcross_NR = nrbu.get_wf_pols(
       simulations.simulations[sim_number]['wavefile'], mass, inclination=inc,
       delta_t=delta_t, f_lower=30.0001, distance=distance)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# APPROXIMANT 
hplus_approx, hcross_approx = get_td_waveform(approximant=approx,
        distance=distance,
        mass1=mass1,
        mass2=mass2,
        spin1x=0.0,
        spin2x=0.0,
        spin1y=0.0,
        spin2y=0.0,
        spin1z=simulations.simulations[sim_number]['spin1z'],
        spin2z=simulations.simulations[sim_number]['spin2z'],
        inclination=inc,
        f_lower=f_low_approx,
        delta_t=delta_t)

hplus_approx = wfutils.taper_timeseries(hplus_approx, 'TAPER_START')
hcross_approx = wfutils.taper_timeseries(hcross_approx, 'TAPER_START')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATCH CALCULATION 

# Make the timeseries consistent lengths
tlen = max([len(hplus_approx), len(hplus_NR), int(datalen/delta_t)])
#tlen = int(datalen / delta_t)

hplus_approx.resize(tlen)
hplus_NR.resize(tlen)
hcross_approx.resize(tlen)
hcross_NR.resize(tlen)


# Interpolate the ASD to the waveform frequencies (this is convenient so that we
# end up with a PSD which overs all frequencies for use in the match calculation
# later
asd = np.interp(hplus_approx.to_frequencyseries().sample_frequencies,
        asd_data[:,0], asd_data[:,1])


# Now insert ASD into a pycbc frequency series so we can use
# pycbc.filter.match() later
noise_psd = pycbc.types.FrequencySeries(asd**2, delta_f =
        hplus_approx.to_frequencyseries().delta_f)


match, _ = pycbc.filter.match(hplus_approx, hplus_NR,
        low_frequency_cutoff=30.0, psd=noise_psd,
        high_frequency_cutoff=upp_bound)


mismatch = 100*(1-match)


# ------------------------------------------------------------------
# DIAGNOSTIC PLOTS
f, ax = pl.subplots(nrows = 1, ncols=1, figsize=(10,5))

print "~~~~~~~~~~~~~~~~~~~~~~~"
print "%s: Mass: %.2f, q: %.2f, a1: %.2f, a2:%.2f, mismatch: %.2f (%%)"%(
        filename, mass, simulations.simulations[sim_number]['q'],
        simulations.simulations[sim_number]['a1'],
        simulations.simulations[sim_number]['a2'], mismatch)

# Normalise to unit SNR
snr_approx_100 = pycbc.filter.sigma(hplus_approx, psd=noise_psd,
        low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
hplus_approx.data[:] /= snr_approx_100

snr_NR_100 = pycbc.filter.sigma(hplus_NR,
        psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
hplus_NR.data[:] /= snr_NR_100


# Fdomain
Hplus_approx = hplus_approx.to_frequencyseries()
Hplus_NR = hplus_NR.to_frequencyseries()
ax.loglog(Hplus_NR.sample_frequencies,
        plot_snr*2*abs(Hplus_NR)*np.sqrt(Hplus_NR.sample_frequencies),
        label='NR')

ax.loglog(Hplus_approx.sample_frequencies,
        plot_snr*2*abs(Hplus_approx)*np.sqrt(Hplus_approx.sample_frequencies),
        label=approx)

ax.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd),
     label='noise psd', color='k', linestyle='--')

ax.set_title("%s: Mass: %.2f, q: %.2f, a1: %.2f, a2:%.2f, mismatch: %.2f (%%)"%(
    filename, mass, simulations.simulations[sim_number]['q'],
        simulations.simulations[sim_number]['a1'],
        simulations.simulations[sim_number]['a2'], mismatch))


ax.legend(loc='upper right')

ax.axvline(30, color='r')
ax.axvline(upp_bound, color='r')

ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('2|H$_+$($f$)|$\sqrt{f}$ & $\sqrt{S(f)}$')
ax.set_ylim(0.01*min(asd), 10*max(asd))
ax.set_xlim(9, 2e3)

f.tight_layout()

f.savefig(filename.replace('h5','png'))

# ------------------------------------------------------------------
#






