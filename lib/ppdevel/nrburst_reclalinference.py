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
nrburst_reclalinference.py

Post-process posterior samples from lalinference to produce the time-domain
reconstructed waveform

"""

import sys, os
from optparse import OptionParser

import numpy as np
from matplotlib import pyplot as pl

import pycbc.types
import pycbc.waveform 
import pycbc.waveform.utils as wfutils
from pycbc.detector import Detector

from pylal import bayespputils as bppu


def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-d", "--delta-t", type=float, default=1./512)
    parser.add_option("-L", "--datalen", type=float, default=4.0)
    parser.add_option("-a", "--approximant", type=str, default=None)
    parser.add_option("-p", "--psds", type=str, default=None)

    (opts,args) = parser.parse_args()


    if len(args)==0:
        print >> sys.stderr, "Must supply a posterior samples file as a commandline argument"
        sys.exit()

    if not os.path.isfile(args[0]):
        print >> sys.stderr, "posterior samples file requested: %s does not exist"%args[0]
        sys.exit()

    return opts, args


def main(opts,args):

    #
    # Check waveform is valid and choose td/fd generator
    #
    td_approximants = pycbc.waveform.td_approximants()
    fd_approximants = pycbc.waveform.fd_approximants()
    if opts.approximant is in td_approximants:
        get_waveform = pycbc.waveform.get_td_waveform
    elif opts.approximant is in fd_approximants:
        get_waveform = pycbc.waveform.get_fd_waveform
    else:
        print >> sys.stderr, "waveform %s not recognised"%(
                opts.approximant)


    #
    # Load and parse posterior samples file
    #
    peparser = bppu.PEOutputParser('common')
    resultsObj = peparser.parse(open(args[0], 'r'))
    posterior = bppu.Posterior(resultsObj)

    #
    # Reconstructions
    #

    # Loop through posterior samples, build waveform(s)
    for s in xrange(len(posterior)):
        # Component masses
        mass1, mass2 = bppu.q2ms(mc,q)

        





    return h_signal, l_signal, hplus, hcross


if __name__ == "__main__":

    opts, args = parser()

    # return Hanford observation, Livingston observation & polarisations
    h_signal, l_signal, hplus, hcross = main(opts,args)



