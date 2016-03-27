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

from pycbc import pnutils
import nrburst_utils as nrbu

from matplotlib import pyplot as pl
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

matches, masses, inclinations, config, simulations = pickle.load(
        open(args[0], 'rb'))


# Label figures according to the pickle file
if opts.user_tag is None:
    user_tag=args[0].replace('.pickle','')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manipulation and derived FOMs
#

for m in matches:
    m[np.isnan(m)]=0.0

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
chieff      = np.zeros(nsimulations_goodmatch)

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

    mass1, mass2 = \
            pnutils.mchirp_eta_to_mass1_mass2(np.median(chirp_masses[s,:]),
                    sim['eta'])
    chieff[s] = pnutils.phenomb_chi(mass1, mass2, sim['spin1z'],
            sim['spin2z'])

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCATTER PLOTS

print >> sys.stdout, "Plotting..."

mass_ratios = 1./mass_ratios

#   f, ax = nrbu.scatter_plot(
#           config,
#           paramx=chieff, paramy=mass_ratios,
#           matches=median_matches, 
#           labely='q',
#           labelx=r'$\chi_{\mathrm{eff}}$')
#
#pl.show()
#sys.exit()

if config.algorithm=='BW' or config.algorithm=='HWINJ':
    clims=[0.5, 1]
elif config.algorithm=='CWB':
    clims=[0.5, 0.9]

f, ax = nrbu.double_scatter_plot(
        config,
        param1x=a1dotL, param2x=a2dotL,
        paramy=mass_ratios,
        matches=median_matches, 
        labely='q',
        label1x=r'$\hat{\mathbf{S}}_1 . \hat{\mathbf{L}}$',
        label2x=r'$\hat{\mathbf{S}}_2 . \hat{\mathbf{L}}$', clims=clims)

ax[0].annotate('# sims: %d'%len(simulations_goodmatch), (-0.9, 1.1))

f.tight_layout()
pl.subplots_adjust(bottom=0.3)
f.savefig("%s_FF_s1dotLs2dotL-q.eps"%user_tag)
f.savefig("%s_FF_s1dotLs2dotL-q.png"%user_tag)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA DUMP
filename="%s_FF_s1dotLs2dotL-q"%user_tag
np.savez(filename, a1dotL=a1dotL, a2dotL=a2dotL, mass_ratios=mass_ratios,
        median_matches=median_matches)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOX PLOTS

Nwaves=25
if config.algorithm=='BW':
    f, ax = nrbu.matchboxes(matches, simulations_goodmatch, Nwaves)
elif config.algorithm=='CWB':
    Nwaves=25
    f, ax = nrbu.matchpoints(matches, simulations_goodmatch, Nwaves)
elif config.algorithm=='HWINJ':
    Nwaves=25
    f, ax = nrbu.matchpoints(matches, simulations_goodmatch, Nwaves)
    ax.set_xlim(0.8,1)
ax.set_title('Top %d ranked waveforms (%s)'%(Nwaves,user_tag))
f.tight_layout()
f.savefig("%s_FF_ranking.eps"%user_tag)
f.savefig("%s_FF_ranking.png"%user_tag)

sys.exit()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform and data plots
best_simulation = simulations_goodmatch[matchsort][-1]
best_mass = median_masses[matchsort][-1]
std_mass = std_masses[matchsort][-1]
best_inclination = np.median(inclinations[matchsort][-1])

# Generate this waveform with this mass and inclianti

hplus, _ = \
        nrbu.get_wf_pols(best_simulation['wavefile'],
        best_mass, inclination=best_inclination, delta_t=config.delta_t)

# Useful time/freq samples
time_axis = np.arange(0, config.datalen, config.delta_t)
hplus.resize(len(time_axis))
freq_axis = np.arange(0.5*config.datalen/config.delta_t+1./config.datalen) * 1./config.datalen

# Interpolate the ASD to the waveform frequencies (this is convenient so that we
# end up with a PSD which overs all frequencies for use in the match calculation
# later - In practice, this will really just pad out the spectrum at low
# frequencies)
h1_asd_data = np.loadtxt(config.h1_spectral_estimate)
l1_asd_data = np.loadtxt(config.l1_spectral_estimate)

h1_asd = np.exp(np.interp(np.log(freq_axis), np.log(h1_asd_data[:,0]),
    np.log(h1_asd_data[:,1])))
l1_asd = np.exp(np.interp(np.log(freq_axis), np.log(l1_asd_data[:,0]),
    np.log(l1_asd_data[:,1])))

# Whiten
h1tilde = hplus.to_frequencyseries()
h1tilde.data /= h1_asd
h1_white = h1tilde.to_timeseries()

l1tilde = hplus.to_frequencyseries()
l1tilde.data /= l1_asd
l1_white = l1tilde.to_timeseries()




