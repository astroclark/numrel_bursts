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
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.waveform import utils as wfutils
import pycbc.filter
from pycbc import pnutils
from pycbc import fft


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

#
# --- Catalogue Definition
#
bounds = dict()

home=os.environ.get('HOME')
data_path=os.path.join(home, 'Projects/numrel_bursts/gatech_data')
# *************************************
#   1)  D12_q2.00_a0.15_-0.60_m200 
bounds['q'] = [1.999, 2.001]
bounds['spin1z'] = [0.14, 0.16]
bounds['spin2z'] = [-0.59, -0.61]

errors_file = os.path.join(data_path,
        'Strain_Lframe_l2_m2_r75_D12_q2.00_a0.15_-0.60_m200.asc')

 
#    2) Sq4_d9_a0.6_oth.270_rr_M180 
#    - Highest resolution of this run missing in h5 catalog
#    - for lower res try: GATECH0410.h5
#   bounds['q'] = [3.9999, 4.0001]
#   bounds['spin1z'] = [0, 0.00001]
#   bounds['spin2z'] = [0, 0.00001]
#   bounds['spin1x'] = [-0.61, -0.59]
#   bounds['spin2x'] = [-0.61, -0.59]
#   errors_file = os.path.join(data_path,
#           'Strain_Lframe_l2_m2_r75_Sq4_d9_a0.6_oth.270_rr_M180.asc')


#    3) D7.5_q15.00_a0.0_CHgEEB_m800
#   bounds['q'] = [14.99, 15.01]
#   bounds['q'] = [10, np.inf]
#   errors_file = os.path.join(data_path,
#           'Strain_Lframe_l2_m2_r75_Sq4_d9_a0.6_oth.270_rr_M180.asc')


#   #    4) RO3_D10_q1.50_a0.60_oth.090_M120
#   bounds['q'] = [1.4999, 1.50001]
#   bounds['spin1z'] = [0, 0.0001]
#   bounds['spin2z'] = [0.59, 0.61]
#   errors_file = os.path.join(data_path,
#   'Strain_Lframe_l2_m2_r75_RO3_D10_q1.50_a0.60_oth.090_M120.asc')

#    5) q8_LL_D9_a0.6_th1_45_th2_225
#bounds['q'] = [7.5, 8.5]
#errors_file = os.path.join(data_path,
#'Strain_Lframe_l2_m2_r75_q8_LL_D9_a0.6_th1_45_th2_225_m400.asc')
#'Strain_Simframe_l2_m2_r75_q8_LL_D9_a0.6_th1_45_th2_225_m400.asc')

# *************************************

inc = 0 
approx='SEOBNRv2'
#approx='IMRPhenomPv2'
#approx='IMRPhenomP'
f_low_approx=20


#
# --- Plotting options
#
nMassPoints = 5
maxMass = 500.0

#
# --- Time Series Config
#
delta_t = 1./4096
datalen = 8

#
# --- Noise Spectrum
#
asd_file = os.path.join(home, 'GW150914_data/noise_curves/early_aligo.dat')

#
# --- Catalog
#
catalog=os.path.join(home, 'GW150914_data/nr_catalog/gatech_hdf5')

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
        nrbu.simulation_details(param_bounds=bounds,
                catdir=catalog)

if simulations.simulations[0]['wavefile'].split('/')[-1] ==  'GATECH1469.h5':
    simulations.simulations[0]['Mmin30Hz'] = 71.0

asd_data = np.loadtxt(asd_file)

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

# Extract the data
times_codeunits, hplus_NR_codeunits, hcross_NR_codeunits, \
        dAmpbyAmp_codeunits, dphi_codeunits = parse_NR_asc(errors_file)

# Set up the Masses we're going to study
masses = np.linspace(simulations.simulations[0]['Mmin30Hz'], maxMass,
        nMassPoints) + 5

# matches is going to be a list of tuples: (mass, match)
matches = []

f, ax = pl.subplots(nrows = len(masses), ncols=2, figsize=(15,15))

for m,mass in enumerate(masses):
    "Extracting and generating mass %d of %d (%.2f)"%(m, len(masses), mass)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Common params

    mass1, mass2 = pnutils.mtotal_eta_to_mass1_mass2(mass,
            simulations.simulations[0]['eta'])


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NUMERICAL RELATIVITY

    # --- Generate the polarisations
#   hplus_NR, hcross_NR = nrbu.get_wf_pols(
#           simulations.simulations[0]['wavefile'], mass, inclination=inc,
#           delta_t=delta_t, f_lower=29 * min(masses) / mass, distance=distance)

    hplus_NR  = scale_NR(times_codeunits, hplus_NR_codeunits, mass,
            delta_t=delta_t)
    hcross_NR = scale_NR(times_codeunits, hcross_NR_codeunits, mass,
            delta_t=delta_t)

    hplus_NR.data = window_wave(hplus_NR.data)
    hcross_NR.data = window_wave(hcross_NR.data)

    hplus_NR = wfutils.taper_timeseries(hplus_NR, 'TAPER_STARTEND')
    hcross_NR = wfutils.taper_timeseries(hcross_NR, 'TAPER_STARTEND')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # APPROXIMANT 

    if approx == 'SEOBNRv2':

        hplus_approx, hcross_approx = get_td_waveform(approximant=approx,
                distance=distance,
                mass1=mass1,
                mass2=mass2,
                spin1x=0.0,
                spin2x=0.0,
                spin1y=0.0,
                spin2y=0.0,
                spin1z=simulations.simulations[0]['spin1z'],
                spin2z=simulations.simulations[0]['spin2z'],
                inclination=inc,
                f_lower=f_low_approx * min(masses)/mass,
                delta_t=delta_t)
                #spin1x=simulations.simulations[0]['spin1x'],
                #spin2x=simulations.simulations[0]['spin2x'],
                #spin1y=simulations.simulations[0]['spin1y'],
                #spin2y=simulations.simulations[0]['spin2y'],
                #spin1z=simulations.simulations[0]['spin1z'],
                #spin2z=simulations.simulations[0]['spin2z'],

        # Make the timeseries consistent lengths
        tlen = max(len(hplus_approx), len(hplus_NR))

        hplus_approx.resize(tlen)
        hplus_NR.resize(tlen)
        hcross_approx.resize(tlen)
        hcross_NR.resize(tlen)

        # Estimate ffinal 
        #ffinal = pnutils.get_final_freq(approx, mass1, mass2, 
        #        simulations.simulations[0]['spin1z'],simulations.simulations[0]['spin2z'])

    elif approx == 'IMRPhenomPv2' or approx == 'IMRPhenomP':

        Hplus_approx, Hcross_approx = get_fd_waveform(approximant=approx,
                distance=distance,
                mass1=mass1,
                mass2=mass2,
                spin1x=simulations.simulations[0]['spin1x'],
                spin2x=simulations.simulations[0]['spin2x'],
                spin1y=simulations.simulations[0]['spin1y'],
                spin2y=simulations.simulations[0]['spin2y'],
                spin1z=simulations.simulations[0]['spin1z'],
                spin2z=simulations.simulations[0]['spin2z'],
                inclination=inc,
                f_lower=f_low_approx * min(masses)/mass,
                delta_f=hplus_NR.to_frequencyseries().delta_f)

        tlen = int(1.0 / hplus_NR.delta_t / Hplus_approx.delta_f)

        Hplus_approx.resize(tlen/2 + 1)
        hplus_approx = pycbc.types.TimeSeries(pycbc.types.zeros(tlen), delta_t=hplus_NR.delta_t)
        fft.ifft(Hplus_approx, hplus_approx)

        Hcross_approx.resize(tlen/2 + 1)
        hcross_approx = pycbc.types.TimeSeries(pycbc.types.zeros(tlen), delta_t=hcross_NR.delta_t)
        fft.ifft(Hcross_approx, hcross_approx)

    hplus_approx = wfutils.taper_timeseries(hplus_approx, 'TAPER_START')
    hcross_approx = wfutils.taper_timeseries(hcross_approx, 'TAPER_START')


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MATCH CALCULATION 
 
    #upp_bound = ffinal
    #upp_bound = 1.5*ffinal
    upp_bound = 0.5/delta_t
    
    # ***********************
    # Errors
    # Convert to amplitude/phase
 
    dAmpbyAmp = scale_NR(times_codeunits, dAmpbyAmp_codeunits, mass, delta_t=delta_t)
    dAmpbyAmp.resize(tlen)
    dphi = scale_NR(times_codeunits, dphi_codeunits, mass, delta_t=delta_t)
    dphi.resize(tlen)
 
    amp_NR = wfutils.amplitude_from_polarizations(hplus_NR, hcross_NR)
    amp_NR_deltaUpp = amp_NR + dAmpbyAmp*amp_NR
    amp_NR_deltaLow = amp_NR - dAmpbyAmp*amp_NR 

    phi_NR = wfutils.phase_from_polarizations(hplus_NR, hcross_NR)
    phi_NR_deltaUpp = phi_NR + dphi
    phi_NR_deltaLow = phi_NR - dphi
 
    hplus_NR_deltaLow = \
            pycbc.types.TimeSeries(np.real(amp_NR_deltaLow*np.exp(1j*phi_NR_deltaLow)),
                    delta_t=hplus_NR.delta_t)
    hplus_NR_deltaUpp = \
            pycbc.types.TimeSeries(np.real(amp_NR_deltaUpp*np.exp(1j*phi_NR_deltaUpp)),
                    delta_t=hplus_NR.delta_t)
 

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

    match_deltaUpp, _ = pycbc.filter.match(hplus_NR, hplus_NR_deltaUpp,
            low_frequency_cutoff=30.0, psd=noise_psd,
            high_frequency_cutoff=upp_bound) 

    match_deltaLow, _ = pycbc.filter.match(hplus_NR, hplus_NR_deltaLow,
            low_frequency_cutoff=30.0, psd=noise_psd,
            high_frequency_cutoff=upp_bound)

    mismatch = 100*(1-match)

    mismatch_deltaUpp = 100*(1-match_deltaUpp)
    mismatch_deltaLow = 100*(1-match_deltaLow)

    mismatch_delta = abs(np.mean([mismatch_deltaUpp, mismatch_deltaLow]))

    # ------------------------------------------------------------------
    # DIAGNOSTIC PLOTS

    print "~~~~~~~~~~~~~~~~~~~~~~~"
    print "Mass: %.2f, mismatch: %.2f +/- %.2e (%%)"%(mass, mismatch,
            mismatch_delta)

    # Normalise to unit SNR
    snr_approx_100 = pycbc.filter.sigma(hplus_approx, psd=noise_psd,
            low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
    hplus_approx.data[:] /= snr_approx_100

    snr_NR_100 = pycbc.filter.sigma(hplus_NR,
            psd=noise_psd, low_frequency_cutoff=30, high_frequency_cutoff=upp_bound)
    hplus_NR.data[:] /= snr_NR_100

    # Tdomain

    maxidx = np.argmax(hplus_NR)
    ax[m][0].plot(hplus_NR.sample_times -
            hplus_NR.sample_times[maxidx], hplus_NR,
            label='NR')

    maxidx = np.argmax(hplus_approx)
    ax[m][0].plot(hplus_approx.sample_times -
            hplus_approx.sample_times[maxidx], hplus_approx,
            label='approx')

    ax[m][0].set_title('M$_{\mathrm{tot}}$=%.2f M$_{\odot}$, mismatch=%.2f +/- %.2e (%%)'%(
        mass, 100*(1-match), mismatch_delta))

    ax[m][0].set_xlabel('Time [s]')
    ax[m][0].set_ylabel('h(t) [arb units]')


#    ax[m][0].set_xlim(-0.25, 0.1)
    ax[m][0].set_xlim(-3, 1)

    # Fdomain
    Hplus_approx = hplus_approx.to_frequencyseries()
    Hplus_NR = hplus_NR.to_frequencyseries()
    ax[m][1].loglog(Hplus_NR.sample_frequencies,
               plot_snr*2*abs(Hplus_NR)*np.sqrt(Hplus_NR.sample_frequencies),
               label='NR')

    ax[m][1].loglog(Hplus_approx.sample_frequencies,
               plot_snr*2*abs(Hplus_approx)*np.sqrt(Hplus_approx.sample_frequencies),
               label=approx)

    ax[m][1].loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd),
            label='noise psd', color='k', linestyle='--')

    ax[m][1].set_title('M$_{\mathrm{tot}}$=%.2f M$_{\odot}$, mismatch=%.2f +/- %.2e (%%)'%(
        mass, 100*(1-match), mismatch_delta))

    ax[m][1].legend(loc='upper right')

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






