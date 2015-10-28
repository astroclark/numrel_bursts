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
nrburst_wf_plots.py

Produces data for the best matching NR waveforms to the reconstruction

"""

import sys, os
import copy
import subprocess
from optparse import OptionParser
import cPickle as pickle
import timeit

import numpy as np
from matplotlib import pyplot as pl
import triangle

import lal
import pycbc.types
from pylal import spawaveform

import burst_nr_utils as nrbu

pl.rcParams.update({'axes.labelsize': 16})
pl.rcParams.update({'xtick.labelsize':16})
pl.rcParams.update({'ytick.labelsize':16})
pl.rcParams.update({'legend.fontsize':16})

def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-i", "--ifo-label", default="Unlabelled IFO", type=str)
    parser.add_option("-t", "--user-tag", type=str, default=None)
    parser.add_option("-m", "--match-threshold", type=float, default=0.0)
    parser.add_option("-c", "--match-lim-high", type=float, default=1.0)
    parser.add_option("-u", "--match-clim-upp", type=float, default=0.95)
    parser.add_option("-l", "--match-clim-low", type=float, default=0.90)
    parser.add_option("-L", "--no-plot", action="store_true", default=False)
    parser.add_option("-a", "--asd-data", type=str)

    (opts,args) = parser.parse_args()

    return opts, args

def make_labels(simulations):
    """
    Return a list of strings with suitable labels for e.g., box plots
    """

    labels=[]
    for sim in simulations:

        # check nans
        vals = []
        for val in [sim['q'], sim['a1'], sim['a2'], sim['th1L'], sim['th2L']]:
            if np.isnan(val):
                val = 0.0
            vals.append(val)

        labelstr = \
                r"$q=%.2f$, $a_1=%.2f$, $a_2=%.2f$, $\theta_1=%.2f$, $\theta_2=%.2f$"%(
                        vals[0], vals[1], vals[2], vals[3], vals[4])
        labels.append(labelstr)

    return labels



__author__ = "James Clark <james.clark@ligo.org>"
#git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
#__version__ = "git id %s" % git_version_id

gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse Input
#
# Trivial: just load the pickle

opts, args = parser()

matches, masses, config, simulations, _, _, _ = pickle.load(
        open(args[0], 'rb'))

for sim in simulations.simulations:
    sim['wavefile'] = sim['wavefile'].replace('jclark308','jclark')

config.spectral_estimate = config.spectral_estimate.replace('jclark308','jclark')

# Label figures according to the pickle file
if opts.user_tag is None:
    user_tag=args[0].strip('.pickle')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manipulation and derived FOMs
#

# Remove zero-match waveforms:
mean_matches = np.mean(matches, axis=1)

nonzero_match = mean_matches>opts.match_threshold
matches = matches[nonzero_match]
masses = masses[nonzero_match]

# XXX: bit hacky..
simulations_goodmatch = np.array(simulations.simulations)[nonzero_match]
nsimulations_goodmatch = len(simulations_goodmatch)


# Continue
mean_matches = np.mean(matches, axis=1)
median_matches = np.median(matches, axis=1)
std_matches = np.std(matches, axis=1)

median_masses = np.median(masses, axis=1)
std_masses = np.std(masses, axis=1)

mass_ratios = np.zeros(nsimulations_goodmatch)
chis = np.zeros(shape=(nsimulations_goodmatch, config.nsampls))
chirp_masses = np.zeros(shape=(nsimulations_goodmatch, config.nsampls))

theta1L = np.zeros(nsimulations_goodmatch)
theta2L = np.zeros(nsimulations_goodmatch)
thetaSL = np.zeros(nsimulations_goodmatch)

for s, sim in enumerate(simulations_goodmatch):

    mass_ratios[s] = sim['q']

    spin1z = nrbu.cartesian_spins(sim['a1'], sim['th1L'])
    spin2z = nrbu.cartesian_spins(sim['a2'], sim['th2L'])

    if np.isnan(sim['th1L']): theta1L[s]=0.0
    else: theta1L[s]=sim['th1L']

    if np.isnan(sim['th2L']): theta2L[s]=0.0
    else: theta2L[s]=sim['th2L']

    if np.isnan(sim['thSL']): theta1L[s]=0.0
    else: thetaSL[s]=sim['thSL']


    for n in xrange(config.nsampls):

        mass1, mass2 = nrbu.component_masses(masses[s, n], mass_ratios[s])

        chirp_masses[s, n] = spawaveform.chirpmass(mass1, mass2) \
                / lal.MTSUN_SI
        chis[s, n] = spawaveform.computechi(mass1, mass2, spin1z, spin2z)

median_chirp_masses = np.median(chirp_masses, axis=1)
std_chirp_masses = np.std(chirp_masses, axis=1)

median_chis = np.median(chis, axis=1)
std_chis = np.std(chis, axis=1)

matchsort = np.argsort(median_matches)
print "~~~~~~~~~~~~~~~~~"
print "Summary for %s"%args[0]
print ""
print "   * Match: %f +/- %f"%(median_matches[matchsort][-1],
        std_matches[matchsort][-1])
print "   * total mass: %f +/- %f"%(median_masses[matchsort][-1],
        std_masses[matchsort][-1])
print "   * chirp mass: %f +/- %f"%(median_chirp_masses[matchsort][-1],
        std_chirp_masses[matchsort][-1])
print "   * eff spin: %f +/- %f"%(median_chis[matchsort][-1], std_chis[matchsort][-1])

print " FULL DETAILS:"
print np.array(simulations.simulations)[matchsort][-1]

if opts.no_plot: sys.exit(0)

# XXX: DANGEROUS HACKING

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform Plots

# --- Create a catalogue with the top ranked NR waveform
# XXX: easily extensible to the top N waveforms

best_sims = copy.deepcopy(simulations)
best_sims.simulations = [np.array(simulations.simulations)[matchsort][-1]]
best_sims.nsimulations = 1

# XXX: Dial up the sample rate for nice smooth waveforms
plot_sample_rate = 1024
init_total_mass = 100.0
catalogue = nrbu.waveform_catalogue(best_sims, ref_mass=init_total_mass,
        SI_deltaT=1./plot_sample_rate, SI_datalen=config.datalen, distance=1.0,
        trunc_time=False)

# Generate the median-mass waveform +/- 1 sigma
wave = pycbc.types.TimeSeries(np.real(catalogue.SIComplexTimeSeries[0]),
        delta_t=1./plot_sample_rate)

# -- Retrieve the spectral estimate
freq_axis = wave.to_frequencyseries().sample_frequencies.data[:]
#asd_data = np.loadtxt(config.spectral_estimate)
#asd = np.interp(freq_axis, asd_data[:,0], asd_data[:,1])

asd_data = np.loadtxt(config.spectral_estimate)
asd = np.interp(freq_axis, asd_data[:,0], asd_data[:,1])


#asd_data = np.loadtxt('/home/jclark/Projects/bhextractor/data/observed/bw/waveforms/clean_psd_199.dat.0')
#asd = np.interp(freq_axis, asd_data[:,0], np.sqrt(asd_data[:,2]/2))


# XXX
#   from pycbc.psd import aLIGOZeroDetHighPower
#   psd = aLIGOZeroDetHighPower(len(wave.to_frequencyseries()),
#           wave.to_frequencyseries().delta_f,
#           wave.to_frequencyseries().sample_frequencies.min()) 
#   asd=np.sqrt(psd.data[:])
#   asd[0] = 1e10

#
# --- Scale the NR data to the median recovered mass
#
median_waveform = pycbc.types.TimeSeries(np.real(nrbu.scale_wave(wave,
    median_masses[matchsort[-1]], init_total_mass)),
    delta_t=1./plot_sample_rate)

# Whitening
median_waveform_white = median_waveform.to_frequencyseries()
median_waveform_white.data[:] /= asd
# Scale to unit hrss
median_waveform_white.data[:] /= pycbc.filter.sigma(median_waveform_white)
median_waveform_white = median_waveform_white.to_timeseries()



#
# --- Scale the NR data to the median recovered mass minus 0.5*stdev
#
low_bound_waveform = pycbc.types.TimeSeries(np.real(nrbu.scale_wave(wave,
    median_masses[matchsort[-1]]-std_masses[matchsort[-1]], init_total_mass) ),
    delta_t=1./plot_sample_rate)

# Whitening
low_bound_waveform_white = low_bound_waveform.to_frequencyseries()
low_bound_waveform_white.data[:] /= asd
low_bound_waveform_white.data[:] /= pycbc.filter.sigma(low_bound_waveform_white)
low_bound_waveform_white = low_bound_waveform_white.to_timeseries()

#
# --- Scale the NR data to the median recovered mass plus 0.5*stdev
#
upp_bound_waveform = pycbc.types.TimeSeries(np.real(nrbu.scale_wave(wave,
    median_masses[matchsort[-1]]+std_masses[matchsort[-1]], init_total_mass) ),
    delta_t=1./plot_sample_rate)

# Whitening
upp_bound_waveform_white = upp_bound_waveform.to_frequencyseries()
upp_bound_waveform_white.data[:] /= asd
upp_bound_waveform_white.data[:] /= pycbc.filter.sigma(upp_bound_waveform_white)
upp_bound_waveform_white = upp_bound_waveform_white.to_timeseries()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting & Conditioning

#   import pycbc.filter
#   medwfhp = pycbc.filter.highpass(median_waveform_white, 16, filter_order=8,
#           attenuation=0.1)
#
#   from scipy import signal
#   b, a = signal.butter(12, 256./(0.5*plot_sample_rate))
#   medlp = signal.filtfilt(b, a, median_waveform_white.data[:])
#   medwflp = pycbc.types.TimeSeries(medlp, delta_t = 1./plot_sample_rate)

from matplotlib import pyplot as pl

pl.figure()
pl.semilogy(median_waveform_white.to_frequencyseries().sample_frequencies,
        abs(median_waveform_white.to_frequencyseries()))
pl.title('abs[FFT] for whitened, IFFTd waveform')
pl.xlim(0, 512)


# XXX


recon = \
        np.loadtxt('/home/jclark/Projects/bhextractor/data/observed/bw/waveforms/signal_recovered_whitened_waveform.dat.1')
median_recon = pycbc.types.TimeSeries(np.median(recon, axis=0), delta_t=1./1024)
median_recon.data /= pycbc.filter.sigma(median_recon)

match, peakidx = pycbc.filter.match(median_recon, median_waveform_white, low_frequency_cutoff=30)

pl.figure()
delta = -4/1024.
pl.plot(median_recon.sample_times-median_recon.sample_times[np.argmax(median_recon)]-delta,
        median_recon, color='k', linewidth=2)
pl.plot(median_waveform_white.sample_times-median_waveform_white.sample_times[np.argmax(median_waveform_white)],
        median_waveform_white, label='IFFT', color='r')
pl.xlim(-0.2,0.1)

#pl.plot(median_waveform_white.sample_times-median_waveform_white.sample_times[np.argmax(median_waveform_white)],
#        medwflp, label='IFFT', color='g')

pl.fill_between(median_waveform_white.sample_times-median_waveform_white.sample_times[np.argmax(median_waveform_white)],
        y1=low_bound_waveform_white, y2=upp_bound_waveform_white, alpha=0.5)

#pl.plot(medwfhp.sample_times-medwfhp.sample_times[np.argmax(medwfhp)],
#        medwfhp, label='highpassed IFFT')

#pl.plot(medwflp.sample_times-medwflp.sample_times[np.argmax(medwflp)],
#        medwflp, label='lowpassed IFFT')

#pl.legend(loc='upper left')

pl.title('IFFT of whitened waveform (match=%.2f)'%(match))


filename=args[0].replace('.pickle', '_NRwave.txt')
outfile = open(filename, 'w')
outfile.writelines('# time_sample median-stdev median median+stdev\n')
for t in xrange(len(median_waveform)):
    outfile.writelines('%f %e %e %e\n'%(
        median_waveform.sample_times[t], low_bound_waveform.data[t],
        median_waveform.data[t], upp_bound_waveform.data[t]))
outfile.close()

filename=args[0].replace('.pickle', '_NRwaveWhite.txt')
outfile = open(filename, 'w')
outfile.writelines('# time_sample median-stdev median median+stdev\n')
for t in xrange(len(median_waveform)):
    outfile.writelines('%f %e %e %e\n'%(
        median_waveform.sample_times[t], low_bound_waveform_white.data[t],
        median_waveform_white.data[t], upp_bound_waveform_white.data[t]))
outfile.close()


# Code below will scale the NR waveform to all masses:

# --- Now create the array of waveforms scaled to the best-fit (smallest
# mismatch) total mass for each waveform sample
#   best_fit_waveforms = np.zeros(shape=(config.nsampls, int(config.datalen *
#       plot_sample_rate)))
#
#   for m, mass_scale in enumerate(masses[matchsort][-1]):
#       print '%d of %d'%(m,len(masses[matchsort][-1])) 
#
#       wave = pycbc.types.TimeSeries(np.real(catalogue.SIComplexTimeSeries[0]),
#               delta_t=1./plot_sample_rate)
#
#       amp, phase = nrbu.scale_wave(wave, mass_scale, init_total_mass)
#
#       best_fit_waveforms[m, :] = np.real(amp*np.exp(1j*phase))
#
#
