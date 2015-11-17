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
import numpy as np
from matplotlib import pyplot as pl

from pycbc import pnutils
import nrburst_utils as nrbu

pl.rcParams.update({'axes.labelsize': 16})
pl.rcParams.update({'xtick.labelsize':16})
pl.rcParams.update({'ytick.labelsize':16})
pl.rcParams.update({'legend.fontsize':16})


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

Mchirpmin30Hz = float(sys.argv[1])
catalog='/home/jclark308/GW150914_data/nr_catalog/gatech_hdf5'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Manipulation and derived FOMs
#

bounds = dict()
bounds['Mchirpmin30Hz'] = [-np.inf, Mchirpmin30Hz]
simulations = nrbu.simulation_details(param_bounds=bounds, catdir=catalog)


# --- Preallocate
mass_ratios = np.zeros(simulations.nsimulations)
sym_mass_ratios = np.zeros(simulations.nsimulations)

a1dotL      = np.zeros(simulations.nsimulations)
a2dotL      = np.zeros(simulations.nsimulations)
a1crossL    = np.zeros(simulations.nsimulations)
a2crossL    = np.zeros(simulations.nsimulations)
SeffdotL    = np.zeros(simulations.nsimulations)
SeffcrossL  = np.zeros(simulations.nsimulations)
theta_a12   = np.zeros(simulations.nsimulations)
SdotL       = np.zeros(simulations.nsimulations)
theta_SdotL = np.zeros(simulations.nsimulations)

for s, sim in enumerate(simulations.simulations):

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCATTER PLOTS

print >> sys.stdout, "Plotting..."

mass_ratios = 1./mass_ratios

f, ax = pl.subplots(ncols=2,sharey=True)
ax[0].scatter(a1dotL, mass_ratios, c='k', marker='x')
ax[0].set_xlabel(r'$\hat{\mathbf{S}}_1 . \hat{\mathbf{L}}$')
ax[0].set_ylabel(r'$q$')
ax[1].scatter(a2dotL, mass_ratios, c='k', marker='x')
ax[1].set_xlabel(r'$\hat{\mathbf{S}}_2 . \hat{\mathbf{L}}$')
#ax[1].set_ylabel(r'$q$')

ax[0].annotate('min M$_{\mathrm{chirp}}$=%.1f $\\rightarrow$ # sims: %d'%(
    Mchirpmin30Hz, simulations.nsimulations), (-0.9, 1.1))

ax[0].minorticks_on()
ax[0].grid()
ax[1].minorticks_on()
ax[1].grid()

f.tight_layout()
#f.savefig("%s_massratio-a1dotLa2dotL.png"%user_tag)


pl.show()
sys.exit()

#   # --- Mass vs SeffdotL
#   f, ax = scatter_plot(param1=median_masses, param2=SeffdotL,
#           matches=median_matches, param1err=std_masses, param2err=None, 
#           label1='Total Mass [M$_{\odot}$]',
#           label2=r'$\hat{S}_{\mathrm{eff}} . \hat{L}$')
#   ax.set_title(user_tag)
#   f.tight_layout()
#   f.savefig("%s_totalmass-SeffdotL.png"%user_tag)
#
# --- mass ratio vs a1.L Scatter plot
f, ax = scatter_plot(param1=mass_ratios, param2=a1dotL,
        matches=median_matches, param1err=None, param2err=None, 
        label1='$q$',
        label2=r'$\hat{\mathrm{S}}_1 . \hat{L}$')
#ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_massratio-a1dotL.png"%user_tag)

# --- mass ratio vs a2.L Scatter plot
f, ax = scatter_plot(param1=mass_ratios, param2=a2dotL,
        matches=median_matches, param1err=None, param2err=None, 
        label1='$q$',
        label2=r'$\hat{\mathrm{S}}_2 . \hat{L}$')
#ax.set_title(user_tag)
f.tight_layout()
f.savefig("%s_massratio-a2dotL.png"%user_tag)


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

Nwaves=10
if config.algorithm=='BW':
    f, ax = matchboxes(matches, simulations_goodmatch, Nwaves)
elif config.algorithm=='CWB':
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
