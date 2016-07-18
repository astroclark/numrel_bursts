#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2016-2017 James Clark <james.clark@ligo.org>
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
nrburst_pickle_bwplot.py
"""
import os,sys
import cPickle as pickle
import timeit
import numpy as np
from matplotlib import pyplot as pl

import pycbc.types
import pycbc.filter

def overlap(wave0,wave1,fmin=16,delta_t=1./1024,norm=True):

    wave0td = pycbc.types.TimeSeries(wave0, delta_t=delta_t)
    wave1td = pycbc.types.TimeSeries(wave1, delta_t=delta_t)

    overlap=pycbc.filter.overlap(wave0td, wave1td, low_frequency_cutoff=fmin,
            normalized=norm)

    return overlap

def whiten(wave, asdarray, delta_t=1./1024):

    wavetd = pycbc.types.TimeSeries(wave, delta_t=delta_t)
    wavefd = wavetd.to_frequencyseries()

    asd=pycbc.types.FrequencySeries(np.zeros(len(wavefd)),
          delta_f=wavefd.delta_f)
    idx = wavefd.sample_frequencies.data >= min(asdarray[:,0])
    asd.data[idx] = asdarray[:,1]
    asd.data[np.invert(idx)]=1.0

    wavefd_white = wavefd/asd

    return wavefd_white.to_timeseries()

#
# Input
#

if len(sys.argv)>=2:
    injfile=sys.argv[1]
    print "loading data from %s"%injfile

    statinfo = os.stat(injfile)
    print "results pickle is %.2f G"%(statinfo.st_size / 1024 / 1024 / 1024.)

    then = timeit.time.time()
    injset = pickle.load(open(injfile,'rb'))
    now = timeit.time.time()
    print "results took %dm, %ds to load"%(divmod(now-then,60))
else:
    print "Using data in environment"

#
# Allocation
#

for var in injset[0].keys():
    vars()[var] = injset[0][var]

nmoments=10001
netoverlaps = np.zeros(shape=(len(injset), nmoments-1))
mynetoverlaps = np.zeros(shape=(len(injset), len(IFO0_whitened_signal)))
netsnr = np.zeros(shape=(len(injset)))
h1snr = np.zeros(shape=(len(injset)))
l1snr = np.zeros(shape=(len(injset)))
snrratio = np.zeros(shape=(len(injset))) 
Zsignal = np.zeros(shape=(len(injset),2))
for i in xrange(len(injset)):
    print "Reading injection %d/%d"%(i+1, len(injset))

    for var in injset[i].keys():
        vars()[var] = injset[i][var]

    #
    # SNR
    #
    h1snr[i] = float(snr[0][1])
    l1snr[i] = float(snr[1][1])
    snrratio[i] = max(h1snr[i]/l1snr[i], l1snr[i]/h1snr[i])
    netsnr[i] = float(snr[2][1])

    #
    # Evidence
    #
    Zsignal[i][0] = float(evidence[2][1])
    Zsignal[i][1] = float(evidence[2][2])

    #
    # Overlaps
    #
    netoverlaps[i,:] = [IFO1_signal_moments[j][-3] for j in
            xrange(1,len(IFO1_signal_moments))]


    #
    # Manual calculation of network overlap (to facilitate different fmin)
    #
    for j in xrange(len(IFO0_whitened_signal)):

        IFO0_whitened_injection = whiten(H1_timeInjection[:,1], IFO0_ASD)
        IFO1_whitened_injection = whiten(L1_timeInjection[:,1], IFO1_ASD)


        ri =  overlap(IFO0_whitened_signal[j], IFO0_whitened_injection, fmin=16,
                norm=False) + overlap(IFO1_whitened_signal[j],
                        IFO1_whitened_injection, fmin=16, norm=False)

        ii =  overlap(IFO0_whitened_injection, IFO0_whitened_injection, fmin=16,
                norm=False) + overlap(IFO1_whitened_injection,
                        IFO1_whitened_injection, fmin=16, norm=False)

        rr =  overlap(IFO0_whitened_signal[j], IFO0_whitened_signal[j], fmin=16,
                norm=False) + overlap(IFO1_whitened_signal[j],
                        IFO1_whitened_signal[j], fmin=16, norm=False)

        mynetoverlaps[i,j] = ri / np.sqrt(ii*rr)


median_overlap = np.array([np.median(netoverlaps[i]) for i in
    xrange(len(injset))])

std_overlap = np.array([np.std(netoverlaps[i]) for i in xrange(len(injset))])

#
# Now clean up and save workspace
#
# This gives us almost all the characteristics we need to compare injection sets
np.savez(file=injfile.replace('.pickle',''), 
        netoverlaps    = netoverlaps,
        mynetoverlaps  = mynetoverlaps,
        netsnr         = netsnr,
        snrratio       = snrratio,
        Zsignal        = Zsignal,
        median_overlap = median_overlap,
        std_overlap    = std_overlap)

