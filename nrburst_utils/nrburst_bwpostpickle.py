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
nrburst_pickle_bwpost.py

Pull out post-processing results rconstructed waveforms, ASDs, overlaps,
evidence for an injection directory and pickle into a dictionary

"""

import os,sys
import cPickle as pickle
import numpy as np
import tarfile

#
# Input
#
tar = tarfile.open(sys.argv[1], 'r:bz2')

parent_directory = os.path.basename(sys.argv[1].replace('.tar.bz2',''))

#
# Retrieve data
#

# members_to_extract: dictionary of key-value pairs to extract from the archive
# where keys will be used to access data in an output dictionary.

members_to_extract = dict()
members_to_extract['evidence']='evidence.dat'
members_to_extract['snr'] = 'snr.txt'
members_to_extract['IFO0_ASD'] = 'IFO0_asd.dat'
members_to_extract['IFO1_ASD'] = 'IFO1_asd.dat'
members_to_extract['IFO0_signal_moments'] = 'post/signal_whitened_moments.dat.0'
members_to_extract['IFO1_signal_moments'] = 'post/signal_whitened_moments.dat.1'
members_to_extract['IFO0_whitened_signal'] = 'post/signal_recovered_whitened_waveform.dat.0'
members_to_extract['IFO1_whitened_signal'] = 'post/signal_recovered_whitened_waveform.dat.1'

injdata=dict()
for member in members_to_extract.keys():
     print 'handling %s'%member

     tardata = tar.extractfile(os.path.join(parent_directory,
        members_to_extract[member]))

     injdata[member] = tardata.readlines()

     if member in ['evidence','snr']:
         injdata[member] = [ val.split() for val in injdata[member] ]

     if member in ['IFO0_ASD','IFO1_ASD']:
         tmp = [ val.split() for val in injdata[member] ]
         injdata[member] = np.zeros(shape=(len(tmp),2))
         for i in xrange(len(tmp)):
             injdata[member][i,0] = tmp[i][0]
             injdata[member][i,1] = tmp[i][1]

     if member in ['IFO0_signal_moments', 'IFO1_signal_moments']:
         injdata[member] = [ val.split() for val in injdata[member] ]
         for i in xrange(1,len(tmp)):
             injdata[member][i] = [ float(val) for val in injdata[member][i] ]
             # So to get e.g., network overlap for first 10 inj:
             # ifo0_overlaps = [IFO0_signal_moments[i][-3] for i in xrange(1,11)]


     if member in ['IFO0_whitened_signal', 'IFO1_whitened_signal']:
         injdata[member] = [ val.split() for val in injdata[member] ]

         for i in xrange(len(injdata[member])):
             injdata[member][i] = [ float(val) for val in injdata[member][i]]
             # To e.g., plot whitened waveforms:
             #for i in xrange(100): plot(IFO0_whitened_signal[i])


#
# Pickle dictionary
#
outfile = parent_directory+'.pickle'
pickle.dump(injdata, open(outfile, 'wb'))




