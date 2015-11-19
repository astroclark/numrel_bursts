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

    (opts,args) = parser.parse_args()

    return opts, args

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse Input
#
# Trivial: just load the pickle

opts, args = parser()

matches, masses, inclinations, config, simulations, _, _, _ = pickle.load(
        open(args[0], 'rb'))


# Label files according to the pickle file
if opts.user_tag is None:
    user_tag=args[0].strip('.pickle')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manipulation and derived FOMs
#

# Remove NR waveforms in which the mean match was less than some threshold
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
SdotL       = np.zeros(config.nsampls)
theta_SdotL = np.zeros(config.nsampls)

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


# Data dump to ascii
header="# match mass chirpmass q eta a1 a2 a1x a1y a1z a2x a2y a2z Seffx Seffy Seffz a1dotL a2dotL a1crossL a2crossL SeffdotL SeffcrossL theta_a12 SdotL theta_SdotL theta_a1 theta_a2\n"
data_dump_file = open("%s_datadump.txt"%user_tag, 'w')
data_dump_file.writelines(header)
for n in xrange(len(simulations_goodmatch)):
    sim = simulations_goodmatch[n]

    S_eff_x = nrbu.S_eff(sim['q'], sim['spin1x'], sim['spin2x'])
    S_eff_y = nrbu.S_eff(sim['q'], sim['spin1y'], sim['spin2y'])
    S_eff_z = nrbu.S_eff(sim['q'], sim['spin1z'], sim['spin2z'])

    th1L = np.arccos(a1dotL[s]/np.linalg.norm(\
            [sim['spin1x'], sim['spin1y'], sim['spin1z']]
            )) / lal.PI_180
    th2L = np.arccos(a2dotL[s]/np.linalg.norm(\
            [sim['spin2x'], sim['spin2y'], sim['spin2z']]
            )) / lal.PI_180

    if np.isnan(th1L):
        th1L=0.0
    if np.isnan(th2L):
        th2L=0.0

    data_dump_file.writelines(
            "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n"%(
                median_matches[s], median_masses[s], median_chirp_masses[s],
                mass_ratios[s], sym_mass_ratios[s], sim['a1'], sim['a2'],
                sim['spin1x'], sim['spin1y'], sim['spin1z'],
                sim['spin2x'], sim['spin2y'], sim['spin2z'],
                S_eff_x, S_eff_y, S_eff_z,
                a1dotL[s], a2dotL[s], a1crossL[s], a2crossL[s], SeffdotL[s],
                SeffcrossL[s], theta_a12[s], SdotL[s], theta_SdotL[s], th1L,
                th2L))

data_dump_file.close()


