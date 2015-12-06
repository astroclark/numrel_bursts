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
nrburst_plots.py

Summary plots for reconstruction analysis

"""

import sys, os
import copy
import subprocess
from optparse import OptionParser
import cPickle as pickle
import lal
import numpy as np
import timeit
from matplotlib import pyplot as pl
import triangle

from pycbc import pnutils
import nrburst_utils as nrbu

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

    (opts,args) = parser.parse_args()

    return opts, args


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse Input
#
# Trivial: just load the pickle

opts, args = parser()

results_files=args.split(',')

for result_file in resutls_files:

    matches, masses, inclinations, config, simulations, _, _, _ = pickle.load(
            open(args[0], 'rb'))


    # Label figures according to the pickle file
    if opts.user_tag is None:
        user_tag=args[0].replace('.pickle','')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Manipulation and derived FOMs
    #

    # Remove NR waveforms in which the mean match was less than some threshold
    mean_matches = np.mean(matches, axis=1)

    nonzero_match = mean_matches>=opts.match_threshold
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

    # --- Preallocate
    mass_ratios = np.zeros(nsimulations_goodmatch)
    sym_mass_ratios = np.zeros(nsimulations_goodmatch)
    chirp_masses = np.zeros(shape=(nsimulations_goodmatch, config.nsampls))

    a1dotL      = np.zeros(nsimulations_goodmatch)
    a2dotL      = np.zeros(nsimulations_goodmatch)
    a1crossL    = np.zeros(nsimulations_goodmatch)
    a2crossL    = np.zeros(nsimulations_goodmatch)
    SeffdotL    = np.zeros(nsimulations_goodmatch)
    SeffcrossL  = np.zeros(nsimulations_goodmatch)
    theta_a12   = np.zeros(nsimulations_goodmatch)
    SdotL       = np.zeros(nsimulations_goodmatch)
    theta_SdotL = np.zeros(nsimulations_goodmatch)

    for s, sim in enumerate(simulations_goodmatch):

        mass_ratios[s] = sim['q']
        sym_mass_ratios[s] = sim['eta']

        a1dotL[s] = sim['a1dotL']
        a2dotL[s] = sim['a2dotL']

        a1crossL[s] = sim['a1crossL']
        a2crossL[s] = sim['a2crossL']

        theta_a12[s] = sim['theta_a12']

        SeffdotL[s] = sim['SeffdotL']
        SeffcrossL[s] = sim['SeffcrossL']

        SdotL[s] = sim['theta_SdotL']
        theta_SdotL[s] = sim['theta_SdotL']

        for n in xrange(config.nsampls):

            chirp_masses[s,n] = masses[s,n] * sim['eta']**(3./5) 

    median_chirp_masses = np.median(chirp_masses, axis=1)
    std_chirp_masses    = np.std(chirp_masses, axis=1)

    matchsort = np.argsort(median_matches)
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Summary for %s"%args[0]
    print ""
    print "   * Highest (median) Match: %f +/- %f"%(median_matches[matchsort][-1],
            std_matches[matchsort][-1])
    print "   * Waveform: %s"%(
            simulations_goodmatch[matchsort][-1]['wavefile'].split('/')[-1])
    print "   * mass ratio: %f"%(mass_ratios[matchsort][-1])
    print "   * total mass: %f +/- %f"%(median_masses[matchsort][-1],
            std_masses[matchsort][-1])
    print "   * chirp mass: %f +/- %f"%(median_chirp_masses[matchsort][-1],
            std_chirp_masses[matchsort][-1])
    print "   * |a1|: %f, |a2|=%f"%(np.around(simulations_goodmatch[matchsort][-1]['a1'],
        decimals=nrbu.__metadata_ndecimals__),
        np.around(simulations_goodmatch[matchsort][-1]['a2'],
            decimals=nrbu.__metadata_ndecimals__))
    print "   * a1.L: %f, a2.L=%f"%(a1dotL[matchsort][-1], a2dotL[matchsort][-1])
    print "   * a1 x L: %f, a2 x L=%f"%(a1crossL[matchsort][-1],
            a2crossL[matchsort][-1])
    print "   * theta12=%f"%(theta_a12[matchsort][-1])
    print "   * S_eff.L=%f"%(SeffdotL[matchsort][-1])
    print "   * |S_eff x L|=%f"%(SeffcrossL[matchsort][-1])
    print "   * S.L=%f"%(SdotL[matchsort][-1])

    f=open("%s_summary.txt"%user_tag, 'w')
    f.writelines("* Highest (median) Match: %f +/- %f\n"%(median_matches[matchsort][-1],
        std_matches[matchsort][-1]))
    f.writelines("* Waveform: %s\n"%(
        simulations_goodmatch[matchsort][-1]['wavefile'].split('/')[-1]))
    f.writelines("* mass ratio: %f\n"%(mass_ratios[matchsort][-1]))
    f.writelines("* total mass: %f +/- %f\n"%(median_masses[matchsort][-1],
            std_masses[matchsort][-1]))
    f.writelines("* chirp mass: %f +/- %f\n"%(median_chirp_masses[matchsort][-1],
            std_chirp_masses[matchsort][-1]))
    f.writelines("* |a1|: %f, |a2|=%f\n"%(
        np.around(simulations_goodmatch[matchsort][-1]['a1'],
            decimals=nrbu.__metadata_ndecimals__),
        np.around(simulations_goodmatch[matchsort][-1]['a2'],
            decimals=nrbu.__metadata_ndecimals__)))
    f.writelines("* a1.L: %f, a2.L=%f\n"%(a1dotL[matchsort][-1],
        a2dotL[matchsort][-1]))
    f.writelines("* a1 x L: %f, a2 x L=%f\n"%(a1crossL[matchsort][-1],
            a2crossL[matchsort][-1]))
    f.writelines("* theta12=%f\n"%(theta_a12[matchsort][-1]))
    f.writelines("* S_eff.L=%f\n"%(SeffdotL[matchsort][-1]))
    f.writelines("* |S_eff x L|=%f\n"%(SeffcrossL[matchsort][-1]))
    f.writelines("* S.L=%f\n"%(SdotL[matchsort][-1]))
    f.close()


if opts.no_plot: sys.exit(0)
