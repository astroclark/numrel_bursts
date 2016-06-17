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
bhex_pca.py
"""

import sys, os
import os.path
import subprocess
import cPickle as pickle

import timeit
import numpy as np

import nrburst_pca_utils as nrbu_pca
import nrburst_utils as nrbu

import warnings
warnings.filterwarnings("ignore")

#
# --- catalog Definition
#
bounds = dict()
bounds['Mchirpmin30Hz'] = [-np.inf, 30.0]
bounds['a1'] = [-0.99, 0.01]
bounds['a2'] = [-0.99, 0.01]

noise_file = \
        '/home/jclark308/Projects/bhextractor/data/noise_curves/early_aligo.dat'

#
# --- Generate initial catalog
#
print >> sys.stdout,  '~~~~~~~~~~~~~~~~~~~~~'
print >> sys.stdout,  'Selecting Simulations'
print >> sys.stdout,  ''
then = timeit.time.time()

# Select simulations
simulations = nrbu.simulation_details(param_bounds=bounds,
        catdir='/home/jclark308/lvc_nr/GaTech')

# Build catalog from HDF5
catalog = nrbu_pca.catalog(simulations, noise_file=noise_file, mtotal=100)

# Peform PCA
bbh_pca = nrbu_pca.bbh_pca(catalog)

# Save PCA data
bbh_pca.file_dump(sys.argv[1])

# Plot freq. evolution
freqs=[]
import pycbc.types
from pycbc.waveform import utils as wfutils
from matplotlib import pyplot as pl

f, ax = pl.subplots(ncols=2,figsize=(12,5))
time = np.arange(0, len(catalog.amplitude_matrix[0,:])/1024., 1./1024)
time -= 0.5*max(time)
for w in xrange(len(simulations.simulations)):
    complex_wave = catalog.amplitude_matrix[w,:] *\
            np.exp(1j*catalog.phase_matrix[w,:])

    hplus = pycbc.types.TimeSeries(np.real(complex_wave), delta_t=1./1024)
    hcross = pycbc.types.TimeSeries(np.imag(complex_wave), delta_t=1./1024)

    freqs.append(wfutils.frequency_from_polarizations(hplus, hcross))

    nonzero=catalog.amplitude_matrix[w,:]>1e-2

    ax[0].plot(time, catalog.amplitude_matrix[w,:],
            label=simulations.simulations[w]['q'])
    ax[1].plot(time[:-1][nonzero], freqs[w][nonzero],
            label=simulations.simulations[w]['q'])

    ax[0].set_xlim(-0.25,0.1)
    ax[1].set_xlim(-0.25,0.1)
    ax[1].set_ylim(5, 256)
    ax[1].axhline(30, color='k', linestyle='--', linewidth=2)

ax[0].set_title('Amplitude')
ax[1].set_title('Frequency')
pl.tight_layout()

f, ax = pl.subplots(ncols=2,nrows=2,figsize=(12,5))
ax[0][0].plot(time, bbh_pca.pca['amplitude_pca'].mean_)
ax[0][0].set_title('mean amplitude')
ax[0][1].set_title('Top 5 PCs')
ax[0][0].set_xlabel('Time [s]')
ax[0][1].set_xlabel('Time [s]')

for n in xrange(5):
    ax[0][1].plot(time, bbh_pca.pca['amplitude_pca'].components_[n,:])
    ax[1][1].plot(time, bbh_pca.pca['phase_pca'].components_[n,:])

ax[1][0].plot(time, bbh_pca.pca['phase_pca'].mean_)
ax[1][0].set_title('mean phase')
ax[1][1].set_title('Top 5 PCs')
ax[1][0].set_xlabel('Time [s]')
ax[1][1].set_xlabel('Time [s]')

pl.tight_layout()

f, ax = pl.subplots()
ax.plot(range(1,len(simulations.simulations)+1), np.real(bbh_pca.matches.T),
        color='grey', alpha=0.5)
ax.set_xlabel('# PCs')
ax.set_ylabel('single IFO (H1) Match')

min_matches = [min(np.real(bbh_pca.matches[:,n])) for n in xrange(33)]
ax.plot(range(1,len(simulations.simulations)+1),min_matches, color='k',
        linestyle='--', linewidth=2)
ax.set_ylim(0.95,1)
ax.minorticks_on()



pl.show()


for w in xrange(33):
    print >> stdout, "-------------------------"
    #print "Waveform: ", simulations.simulations[b]
    print >> stdout,  "Mass ratio: %.2f"%simulations.simulations[w]['q']
    print >> stdout,  "Top 5 Amplitude coefficients: ", bbh_pca.amplitude_betas[w][:5]
    print >> stdout,  "Top 5 Phase coefficients: ", bbh_pca.phase_betas[w][:5]





