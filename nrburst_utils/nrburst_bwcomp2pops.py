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

def parse_filename(filename, netsnr=None):


    components = filename.split('-')

    injected={}
    injected['waveform'] = components[2].replace('pseudoFourPN','')
    injected['mass_ratio'] = float(components[3].split('_')[1])
    injected['iota'] = float(components[4].split('_')[1])
    if components[-1] == '0noise': injected['noise']=False
    else: injected['noise']=True
    injected['netsnr'] = netsnr

    return injected

#
# Input
#
injfiles = sys.argv[1].split(',')

#
# Load data
#

results=[]
injected=[]
for i,injfile in enumerate(injfiles):

    results.append(np.load(injfile))
    netsnr = results[i]['netsnr'][0]
    injected.append(parse_filename(injfile, netsnr))

#
# Plots
#

# --- Overlap distributions
f, ax = pl.subplots()
medianoverlaps = []
for i in xrange(len(injfiles)):
    medianoverlaps.append(np.median(results[i]['mynetoverlaps'], axis=1))
    ax.hist(medianoverlaps[i], normed=True, histtype='step', cumulative=True,
            bins=100)

# Get p-value for KS test between first two distributions
if len(injfiles)==2:
    from scipy import stats
    _, p=stats.ks_2samp(medianoverlaps[0],medianoverlaps[1])
    ax.set_title('KS-test p-value: %.2f'%p)
else:
    ax.set_title('Median overlaps across injection populations')

ax.axhline(0.5, color='k', linewidth=2, linestyle='--', label='Median overlap')
ax.minorticks_on()
ax.set_ylim(0,1)
ax.set_xlabel('Median Network overlap')
ax.set_ylabel('CDF [over injection population]')

pl.show()

# Save command for reduced moments:

#   np.savez(file=outname, 
#           netoverlaps    = netoverlaps,
#           mynetoverlaps  = mynetoverlaps,
#           netsnr         = netsnr,
#           snrratio       = snrratio,
#           Zsignal        = Zsignal,
#           median_overlap = median_overlap,
#           std_overlap    = std_overlap)






