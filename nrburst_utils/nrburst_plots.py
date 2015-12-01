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
    #scat_all.set_clim(opts.match_clim_low,max(matches))
    scat_all.set_clim(min(matches),max(matches))

    colbar = f.colorbar(scat_all) 
    colbar.set_label('FF')

    ax.minorticks_on()

    ax.grid()

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    f.tight_layout()

    return f, ax

def fancy_scatter_plot(param1x, param2x, paramy, label1x='x', label2x='y',
        labely='Y', matches=None):
    """
    Make a scatter plot of param1 against param2, coloured by match value
    """

    match_sort = np.argsort(matches)

    f, ax = pl.subplots(ncols=2, sharey=True)

    cm = pl.cm.get_cmap('gnuplot')

    # Here's a bunch of messing around to get the best matches plotted on top
    scat_all = ax[0].scatter(param1x[match_sort], paramy[match_sort],
        c=matches[match_sort], s=50, alpha=1, cmap=cm)
    for p in match_sort:
        scat_indi = ax[0].scatter(param1x[p], paramy[p], c=matches[p], s=50,
                alpha=1, zorder=matches[p])
    scat_all.set_clim(min(matches),max(matches))


    ax[0].minorticks_on()
    ax[0].grid()
    ax[0].set_xlabel(label1x)
    ax[0].set_ylabel(labely)

    # --- 2nd axis
    scat_all = ax[1].scatter(param2x[match_sort], paramy[match_sort],
        c=matches[match_sort], s=50, alpha=1, cmap=cm)
    for p in match_sort:
        scat_indi = ax[1].scatter(param2x[p], paramy[p], c=matches[p], s=50,
                alpha=1, label='Median', zorder=matches[p])
    scat_all.set_clim(min(matches),max(matches))

    ax[1].minorticks_on()
    ax[1].grid()
    ax[1].set_xlabel(label2x)


    cbar_ax = f.add_axes([0.13, 0.125, 0.8, 0.05])
    colbar = f.colorbar(scat_all, orientation='horizontal', cax=cbar_ax) 
    if config.algorithm=='BW':
        colbar.set_label('Median Fit Factor')
    else:
        colbar.set_label('Fit Factor')


    return f, ax

def make_labels(simulations):
    """
    Return a list of strings with suitable labels for e.g., box plots
    """

    labels=[]
    for s,sim in enumerate(simulations):

        SdotL = sim['SdotL']
        theta_SdotL = sim['theta_SdotL']
        theta_a12 = sim['theta_a12']

        labelstr = \
                r"$q=%.2f$, $a_1=%.2f$, $a_2=%.2f$, $\theta_{1,2}=%.2f$, $\theta_{\mathrm{\hat{S},\hat{L}}}=%.2f$"%(
                        sim['q'], sim['a1'], sim['a2'], theta_a12, theta_SdotL)
        labels.append(labelstr)

    return labels


def matchboxes(matches, simulations, Nwaves):
    """
    Build a (hideous) box plot to show individual waveform match results from
    BayesWave.  Since we're optimising over mass, this is fitting-factor.
    """

    # Find the sorting to present highest matches first.  Sort on median of the
    # match distribution
    match_sort = np.argsort(np.median(matches, axis=1))

    # --- Match vs Waveform boxes
    f, ax = pl.subplots(figsize=(12,16))
    match_box = ax.boxplot(matches[match_sort].T, whis='range', showcaps=True,
            showmeans=True, showfliers=False,
            vert=False)
    ax.set_xlabel('Fitting Factor')
    ax.set_ylabel('Waveform Parameters')
    ax.grid(linestyle='-', color='grey')
    ax.minorticks_on()

    ax.set_ylim(len(mean_matches)-(Nwaves+0.5), len(mean_matches)+0.5)

    ax.set_xlim(0.8,1.0)

    ylabels=make_labels(np.array(simulations)[match_sort])
    ax.set_yticklabels(ylabels)#, rotation=90)

    f.tight_layout()

    return f, ax

def matchpoints(matches, simulations, Nwaves):
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
    ax.set_ylim(len(mean_matches)-(Nwaves+0.5), len(mean_matches)+0.5)
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

matches, masses, inclinations, config, simulations, _, _, _ = pickle.load(
        open(args[0], 'rb'))


# Label figures according to the pickle file
if opts.user_tag is None:
    user_tag=args[0].strip('.pickle')

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


if opts.no_plot: sys.exit(0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCATTER PLOTS

print >> sys.stdout, "Plotting..."

mass_ratios = 1./mass_ratios

# XXX: EXPERIMENTAL
f, ax = fancy_scatter_plot(
        param1x=a1dotL, param2x=a2dotL,
        paramy=mass_ratios,
        matches=median_matches, 
        labely='q',
        label1x=r'$\hat{\mathbf{S}}_1 . \hat{\mathbf{L}}$',
        label2x=r'$\hat{\mathbf{S}}_2 . \hat{\mathbf{L}}$')

ax[0].annotate('# sims: %d'%len(simulations_goodmatch), (-0.9, 1.1))

f.tight_layout()
pl.subplots_adjust(bottom=0.3)
f.savefig("%s_massratio-a1dotLa2dotL.png"%user_tag)


#pl.show()
#sys.exit()

#   # --- Mass vs SeffdotL
#   f, ax = scatter_plot(param1=median_masses, param2=SeffdotL,
#           matches=median_matches, param1err=std_masses, param2err=None, 
#           label1='Total Mass [M$_{\odot}$]',
#           label2=r'$\hat{S}_{\mathrm{eff}} . \hat{L}$')
#   ax.set_title(user_tag)
#   f.tight_layout()
#   f.savefig("%s_totalmass-SeffdotL.png"%user_tag)
#
#   # --- mass ratio vs a1.L Scatter plot
#   f, ax = scatter_plot(param1=mass_ratios, param2=a1dotL,
#           matches=median_matches, param1err=None, param2err=None, 
#           label1='$q$',
#           label2=r'$\hat{\mathrm{S}}_1 . \hat{L}$')
#   #ax.set_title(user_tag)
#   f.tight_layout()
#   f.savefig("%s_massratio-a1dotL.png"%user_tag)
#
#   # --- mass ratio vs a2.L Scatter plot
#   f, ax = scatter_plot(param1=mass_ratios, param2=a2dotL,
#           matches=median_matches, param1err=None, param2err=None, 
#           label1='$q$',
#           label2=r'$\hat{\mathrm{S}}_2 . \hat{L}$')
#   #ax.set_title(user_tag)
#   f.tight_layout()
#   f.savefig("%s_massratio-a2dotL.png"%user_tag)


#   # --- a1.L vs a2.L Scatter plot
#   f, ax = scatter_plot(param1=a1dotL, param2=a2dotL,
#           matches=median_matches, param1err=None, param2err=None, 
#           label1=r'$\hat{a}_1 . \hat{L}$',
#           label2=r'$\hat{a}_2 . \hat{L}$')
#   ax.set_title(user_tag)
#   f.tight_layout()
#   f.savefig("%s_a1dotL-a2dotL.png"%user_tag)

#   # --- a1.L vs |a1xL| Scatter plot
#   f, ax = scatter_plot(param1=a1crossL, param2=a1dotL,
#           matches=median_matches, param1err=None, param2err=None, 
#           label1=r'|$\mathbf{S}_1 \times \hat{L}|$',
#           label2=r'$\mathbf{S}_1 . \hat{L}$')
#   ax.set_title(user_tag)
#   f.tight_layout()
#
#   f, ax = scatter_plot(param1=a2crossL, param2=a2dotL,
#           matches=median_matches, param1err=None, param2err=None, 
#           label1=r'$|\mathbf{S}_2 \times \hat{L}|$',
#           label2=r'$\mathbf{S}_2 . \hat{L}$')
#   ax.set_title(user_tag)
#   f.tight_layout()

#f.savefig("%s_a1dotL-a2dotL.png"%user_tag)


#   # --- Chirp Mass vs Mass ratio Scatter plot
#   f, ax = scatter_plot(param1=median_chirp_masses, param2=mass_ratios,
#           matches=median_matches, param1err=std_chirp_masses, param2err=None,
#           label1='$\mathcal{M}_{\mathrm{chirp}}$ [M$_{\odot}$]',
#           label2='Mass ratio (q=m$_1$/m$_2$)')
#   ax.set_title(user_tag)
#   f.tight_layout()
#   f.savefig("%s_chirpmass-massratio.png"%user_tag)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOX PLOTS

Nwaves=50
if config.algorithm=='BW':
    f, ax = matchboxes(matches, simulations_goodmatch, Nwaves)
elif config.algorithm=='CWB':
    Nwaves=25
    f, ax = matchpoints(matches, simulations_goodmatch, Nwaves)
ax.set_title('Top %d ranked waveforms (%s)'%(Nwaves,user_tag))
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
