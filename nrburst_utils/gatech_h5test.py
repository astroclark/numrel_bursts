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
import subprocess

import numpy as np

import lal
from pylal import spawaveform
import pycbc.types
from pycbc.waveform import get_td_waveform
from pycbc.waveform import utils as wfutils
from pycbc import pnutils
from pycbc.detector import Detector

import pycbc.filter

import h5py

import nrburst_utils as nrbu

from matplotlib import pyplot as pl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse input

#
# --- Time Series Config
#
deltaT = 1./16384
f_lower = 30.0
detector_name = "H1"

import timeit

then = timeit.time.time()

file = '/home/jclark308/GW150914_data/nr_catalog/gatech_hdf5/GATECH0006.h5'

f = h5py.File(file, 'r')

params = {}

# Metadata parameters:
params['mtotal'] = float(sys.argv[1])

params['eta'] = f.attrs['eta']

params['mass1'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[0]
params['mass2'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[1]

params['spin1x'] = f.attrs['spin1x']
params['spin1y'] = f.attrs['spin1y']
params['spin1z'] = f.attrs['spin1z']
params['spin2x'] = f.attrs['spin2x']
params['spin2y'] = f.attrs['spin2y']
params['spin2z'] = f.attrs['spin2z']

params['coa_phase'] = f.attrs['coa_phase']

f.close()

hp, hc = get_td_waveform(approximant='NR_hdf5_pycbc', 
                                 numrel_data=file,
                                 mass1=params['mass1'],
                                 mass2=params['mass2'],
                                 spin1x=params['spin1x'],
                                 spin1y=params['spin1y'],
                                 spin1z=params['spin1z'],
                                 spin2x=params['spin2x'],
                                 spin2y=params['spin2y'],
                                 spin2z=params['spin2z'],
                                 delta_t=deltaT,
                                 f_lower=f_lower,
                                 inclination=float(sys.argv[2]),
                                 coa_phase=params['coa_phase'],
                                 distance=100)


hp_tapered = wfutils.taper_timeseries(hp, 'TAPER_START')
hc_tapered = wfutils.taper_timeseries(hc, 'TAPER_START')

hp_tapered.data[:] /= pycbc.filter.sigma(hp_tapered)
hc_tapered.data[:] /= pycbc.filter.sigma(hc_tapered)

now = timeit.time.time()

print "took %f sec to extract pols"%(now-then)

# Generate the signal in the detector
detector = Detector(detector_name)

#longitude=float(sys.argv[1])
#latitude=float(sys.argv[2])
polarization=float(sys.argv[3])

# MAP from LALINF
latitude=0.0#-1.26907692672 * 180 / np.pi
longitude=0.0#0.80486732096 * 180 / np.pi

signal = detector.project_wave(hp_tapered, hc_tapered, longitude,
        latitude, polarization)
signal.data[:] /= pycbc.filter.sigma(signal)

tlen = max(len(hp_tapered), len(signal))
hp_tapered.resize(tlen+1)
hc_tapered.resize(tlen+1)
signal.resize(tlen+1)

freq_axis = signal.to_frequencyseries().sample_frequencies.data[:]

asd_data = \
        np.loadtxt('/home/jclark/Projects/GW150914_data/bw_reconstructions/GW150914/IFO0_asd.dat')

asd = np.interp(freq_axis, asd_data[:,0], asd_data[:,1])
psd=pycbc.types.FrequencySeries(asd**2, delta_f=np.diff(freq_axis)[0])

match, _ = pycbc.filter.match(signal, hp_tapered, low_frequency_cutoff=30,
        psd=pycbc.types.FrequencySeries(asd**2, delta_f=np.diff(freq_axis)[0]))

print 'match=%f'%match

f, ax = pl.subplots()
ax.plot(signal.sample_times, hp_tapered, label='h$_+$')
ax.plot(signal.sample_times, hc_tapered, label=r'h$_{\times}$')
ax.plot(signal.sample_times, signal, label='response')
ax.set_title('%f %f %f: match=%f'%(latitude, longitude, polarization, match))
ax.legend(loc='upper left')
ax.set_xlim(-1.,0.1)
pl.show()

