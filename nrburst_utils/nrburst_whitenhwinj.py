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
nrburst_whitenhwinj.py

Read in a hardware injection ascii dump and whiten by a BW PSD estimate.  Dumps
the whitened hwinj back out which we can then pretend is a reconstructed
waveform.
"""

import sys, os
import numpy as np
import pycbc.types

# ------------------
# MAIN

#
# Load data
#
strain_data = np.loadtxt(sys.argv[1])
asd_data = np.loadtxt(sys.argv[2])
sample_rate = 1024.0

#
# Insert into pycbc objects and do the whitening
#
strain_time = pycbc.types.TimeSeries(strain_data, delta_t=1./sample_rate)
strain_freq = strain_time.to_frequencyseries()

asd = np.interp(strain_freq.sample_frequencies, asd_data[:,0], asd_data[:,1])
strain_freq_white = pycbc.types.FrequencySeries(strain_freq.data[:]/asd,
        delta_f=strain_freq.delta_f)

strain_time_white = strain_freq_white.to_timeseries()

#
# Dump back out
#
np.savetxt(sys.argv[1].replace('.txt','_WHITENED.txt'),
        strain_time_white.data[:])
