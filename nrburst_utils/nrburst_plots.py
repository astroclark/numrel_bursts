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
from pylal import spawaveform
import numpy as np
import timeit
from matplotlib import pyplot as pl
import triangle

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

def make_labels(simulations):
    """
    Return a list of strings with suitable labels for e.g., box plots
    """

    labels=[]
    for sim in simulations:

        # check nans
        vals = []
        for val in [sim['q'], sim['a1'], sim['a2'], sim['th1L'], sim['th2L']]:
            if np.isnan(val):
                val = 0.0
            vals.append(val)

        labelstr = \
                r"$q=%.2f$, $a_1=%.2f$, $a_2=%.2f$, $\theta_1=%.2f$, $\theta_2=%.2f$"%(
                        vals[0], vals[1], vals[2], vals[3], vals[4])
        labels.append(labelstr)

    return labels

def scatter_plot(param1, param2, matches, param1err=None, param2err=None,
        label1='x', label2='y'):
    """
    Make a scatter plot of param1 against param2, coloured by match value
    """

    match_sort = np.argsort(matches)

    f, ax = pl.subplots()

    err = ax.errorbar(param1, param2, xerr=param1err, yerr=param2err, color='k',
            linestyle='None', label='1$\sigma$', ecolor='grey', zorder=-1)

    cm = pl.cm.get_cmap('gnuplot')

    # Here's a bunch of messing around to get the best matches plotted on top
    scat_all = ax.scatter(param1[match_sort], param2[match_sort],
        c=matches[match_sort], s=50, alpha=1, cmap=cm)

            
    for p in match_sort:
        scat_indi = ax.scatter(param1[p], param2[p], c=matches[p], s=50,
                alpha=1, label='Median', zorder=matches[p])

    #scat_all.set_clim(opts.match_clim_low,opts.match_clim_upp)
    scat_all.set_clim(opts.match_clim_low,max(matches))

    colbar = f.colorbar(scat_all) 
    colbar.set_label('FF')

    ax.minorticks_on()

    ax.grid()

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    f.tight_layout()

    return f, ax

def matchboxes(matches, simulations):
    """
    Build a (hideous) box plot to show individual waveform match results from
    BayesWave.  Since we're optimising over mass, this is fitting-factor.
    """

    # Find the sorting to present highest matches first.  Sort on median of the
    # match distribution
    match_sort = np.argsort(np.median(matches, axis=1))

    # --- Match vs Waveform boxes
    f, ax = pl.subplots(figsize=(12,8))
    match_box = ax.boxplot(matches[match_sort].T, whis='range', showcaps=True,
            showmeans=True, showfliers=False,
            vert=False)
    ax.set_xlabel('Fitting Factor')
    ax.set_ylabel('Waveform Parameters')
    ax.grid(linestyle='-', color='grey')
    ax.minorticks_on()

    ax.set_ylim(len(mean_matches)-25.5, len(mean_matches)+0.5)

    ax.set_xlim(0.8,1.0)

    ylabels=make_labels(np.array(simulations)[match_sort])
    ax.set_yticklabels(ylabels)#, rotation=90)

    f.tight_layout()

    return f, ax

def matchpoints(matches, simulations):
    """
    Build a plot to show individual waveform match results from
    CWB.  Since we're optimising over mass, this is fitting-factor.
    """

    # Find the sorting to present highest matches first.  Sort on median of the
    # match distribution
    matches = np.concatenate(matches)
    match_sort = np.argsort(matches)

    # --- Match vs Waveform boxes
    f, ax = pl.subplots(figsize=(12,8))

    yvals = range(len(matches))[::-1]
    match_plot = ax.plot(matches[match_sort].T, xrange(len(matches)),
            marker='s', color='k', linestyle='None')

    ax.set_xlabel('Fitting Factor')
    ax.set_ylabel('Waveform Parameters')
    ax.grid(linestyle='-', color='grey')
    ax.minorticks_on()

    ax.set_yticks(xrange(len(matches)))
    ax.set_ylim(len(matches)-25.5, len(matches)-0.5)
    ax.set_xlim(0.85,0.95)

    ylabels=make_labels(np.array(simulations)[match_sort])
    ax.set_yticklabels(ylabels)#, rotation=90)

    f.tight_layout()

    return f, ax


__author__ = "James Clark <james.clark@ligo.org>"
gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# Get the current git version
git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(sys.argv[0])).strip()
__version__ = "git id %s" % git_version_id



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse Input
#
# Trivial: just load the pickle

opts, args = parser()

matches, masses, config, simulations, _, _, _ = pickle.load(
        open(args[0], 'rb'))


# Label figures according to the pickle file
if opts.user_tag is None:
    user_tag=args[0].strip('.pickle')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manipulation and derived FOMs
#

# Remove zero-match waveforms:
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

mass_ratios = np.zeros(nsimulations_goodmatch)
chis = np.zeros(shape=(nsimulations_goodmatch, config.nsampls))
chirp_masses = np.zeros(shape=(nsimulations_goodmatch, config.nsampls))

theta1L = np.zeros(nsimulations_goodmatch)
theta2L = np.zeros(nsimulations_goodmatch)
thetaSL = np.zeros(nsimulations_goodmatch)

for s, sim in enumerate(simulations_goodmatch):

    mass_ratios[s] = sim['q']

    spin1z = nrbu.cartesian_spins(sim['a1'], sim['th1L'])
    spin2z = nrbu.cartesian_spins(sim['a2'], sim['th2L'])

    if np.isnan(sim['th1L']): theta1L[s]=0.0
    else: theta1L[s]=sim['th1L']

    if np.isnan(sim['th2L']): theta2L[s]=0.0
    else: theta2L[s]=sim['th2L']

    if np.isnan(sim['thSL']): theta1L[s]=0.0
    else: thetaSL[s]=sim['thSL']


    for n in xrange(config.nsampls):

        mass1, mass2 = nrbu.component_masses(masses[s, n], mass_ratios[s])

        chirp_masses[s, n] = spawaveform.chirpmass(mass1, mass2) \
                / lal.MTSUN_SI
        chis[s, n] = spawaveform.computechi(mass1, mass2, spin1z, spin2z)

median_chirp_masses = np.median(chirp_masses, axis=1)
std_chirp_masses = np.std(chirp_masses, axis=1)

median_chis = np.median(chis, axis=1)
std_chis = np.std(chis, axis=1)

matchsort = np.argsort(median_matches)
print "~~~~~~~~~~~~~~~~~"
print "Summary for %s"%args[0]
print ""
print "   * Match: %f +/- %f"%(median_matches[matchsort][-1],
        std_matches[matchsort][-1])
print "   * total mass: %f +/- %f"%(median_masses[matchsort][-1],
        std_masses[matchsort][-1])
print "   * chirp mass: %f +/- %f"%(median_chirp_masses[matchsort][-1],
        std_chirp_masses[matchsort][-1])
print "   * eff spin: %f +/- %f"%(median_chis[matchsort][-1], std_chis[matchsort][-1])

if opts.no_plot: sys.exit(0)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCATTER PLOTS

# --- Mass vs thSL Scatter plot
f, ax = scatter_plot(param1=median_masses, param2=thetaSL,
        matches=median_matches, param1err=std_masses, param2err=None, 
        label1='Total Mass [M$_{\odot}$]',
        label2=r'$\theta_{\mathrm{S,L}}$ ($\hat{S}_{\mathrm{tot}} . \hat{L}$) [deg]')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_totalmass-thetaSL.png"%user_tag)


# --- theta1 vs theta2 Scatter plot
f, ax = scatter_plot(param1=theta1L, param2=theta2L,
        matches=median_matches, param1err=None, param2err=None, 
        label1=r'$\theta_1$, $\hat{s}_1.\hat{L}$ [deg]',
        label2=r'$\theta_2$, $\hat{s}_2.\hat{L}$ [deg]')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_theta1-theta2.png"%user_tag)


# --- Mass-ratio vs MassScatter plot
f, ax = scatter_plot(param1=mass_ratios, param2=median_masses,
        matches=median_matches, param1err=None, param2err=std_masses, 
        label1='Mass ratio (q=m$_1$/m$_2$)',
        label2='Total Mass [M$_{\odot}$]')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_massratio-totalmass.png"%user_tag)

# --- Mass-ratio vs Chirp MassScatter plot
f, ax = scatter_plot(param1=mass_ratios, param2=median_chirp_masses,
        matches=median_matches, param1err=None, param2err=std_chirp_masses,
        label1='Mass ratio (q=m$_1$/m$_2$)',
        label2='$\mathcal{M}_{\mathrm{chirp}}$ [M$_{\odot}$]')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_massratio-chirpmass.png"%user_tag)


# --- Mass-ratio vs Chi Scatter plot
f, ax = scatter_plot(param1=mass_ratios, param2=median_chis,
        matches=median_matches, param1err=None, param2err=std_chis,
        label1='Mass ratio (q=m$_1$/m$_2$)',
        label2='Effective Spin ($\chi$)')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_massratio-chi.png"%user_tag)


# --- Mass vs Chi Scatter plot
f, ax = scatter_plot(param1=median_masses, param2=median_chis,
        matches=median_matches, param1err=std_masses, param2err=std_chis,
        label1='Total Mass [M$_{\odot}$]',
        label2='Effective Spin ($\chi$)')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_totalmass-chi.png"%user_tag)

# --- Chirp Mass vs Chi Scatter plot
f, ax = scatter_plot(param1=median_chirp_masses, param2=median_chis,
        matches=median_matches, param1err=std_chirp_masses, param2err=std_chis,
        label1='$\mathcal{M}_{\mathrm{chirp}}$ [M$_{\odot}$]',
        label2='Effective Spin ($\chi$)')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_chirpmass-chi.png"%user_tag)


# --- Chirp Mass vs Total Scatter plot
f, ax = scatter_plot(param1=median_chirp_masses, param2=median_masses,
        matches=median_matches, param1err=std_chirp_masses, param2err=std_masses,
        label1='$\mathcal{M}_{\mathrm{chirp}}$ [M$_{\odot}$]',
        label2='Total Mass [M$_{\odot}$]')
ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_totalmass-chirpmass.png"%user_tag)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOX PLOTS

if config.algorithm=='BW':
    f, ax = matchboxes(matches, simulations_goodmatch)
    ax.set_title('Top 25 ranked waveforms (%s)'%user_tag)
    f.tight_layout()
    f.savefig("%s_matchranking.png"%user_tag)
elif config.algorithm=='CWB':
    f, ax = matchpoints(matches, simulations_goodmatch)
    ax.set_title('Top 25 ranked waveforms (%s)'%user_tag)
    f.tight_layout()
    f.savefig("%s_matchranking.png"%user_tag)

#pl.show()

#
#   samples = np.array([matches[match_sort[-1],:], masses[match_sort[-1],:]]).T
#   trifig = triangle.corner(samples, quantiles=[0.25, 0.50, 0.75], labels=['Match', 
#       'M$_{\mathrm{tot}}$ [M$_{\odot}$]'], plot_contours=True,
#       plot_datapoints=True)
#   title = make_labels([simulations.simulations[match_sort[-1]]])
#   trifig.suptitle(title[0], fontsize=16)
#   trifig.subplots_adjust(top=0.9)
#
#   pl.show()
#
