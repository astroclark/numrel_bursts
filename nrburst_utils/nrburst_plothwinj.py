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
import numpy as np
from matplotlib import pyplot as pl


pl.rcParams.update({'axes.labelsize': 16})
pl.rcParams.update({'xtick.labelsize':16})
pl.rcParams.update({'ytick.labelsize':16})
pl.rcParams.update({'legend.fontsize':16})

raw_result=np.load('/home/jclark/GW150914_data/hwinj_dumps/HWINJ.npz')
bw_result=np.load('/home/jclark/GW150914_data/nrburst_analysis/injection_waveforms_071215/BW.npz')
cwb_result=np.load('/home/jclark/GW150914_data/nrburst_analysis/cwb_injection_waveforms_071215/CWB.npz')

labels=['injection data', 'BayesWave', 'CWB']

#
# Fitting factor vs Total Mass
#
f_mtot, ax_mtot = pl.subplots()


ax_mtot.plot(raw_result['best_match'],
        raw_result['injected_mass']-raw_result['best_mass'],
        linestyle='none', color='k', marker='s', markerfacecolor='none',
        label="Injection Data")

ax_mtot.errorbar(bw_result['best_match'],
        bw_result['injected_mass']-bw_result['best_mass'],
        yerr=bw_result['sigma_best_mass'], xerr=bw_result['sigma_best_match'],
        linestyle='none', color='k', marker='o', label="BayesWave")

ax_mtot.plot(cwb_result['best_match'],
        cwb_result['injected_mass']-cwb_result['best_mass'],
        linestyle='none', color='k', marker='s', label="CWB")

ax_mtot.minorticks_on()
ax_mtot.set_xlabel('Fitting Factor')
ax_mtot.set_ylabel('Injected - Recovered Total Mass [M$_{\odot}$]')
ax_mtot.legend(loc='upper left')
f_mtot.tight_layout()

f_mtot.savefig('totalmass_fittingfactor.eps')

#
# Fitting factor vs Total Mass
#
f_mchirp, ax_mchirp = pl.subplots()


ax_mchirp.plot(raw_result['best_match'],
        raw_result['injected_chirp_mass']-raw_result['best_chirp_mass'],
        linestyle='none', color='k', marker='s', markerfacecolor='none',
        label="Injection Data")

ax_mchirp.errorbar(bw_result['best_match'],
        bw_result['injected_chirp_mass']-bw_result['best_chirp_mass'],
        yerr=bw_result['sigma_best_chirp_mass'], xerr=bw_result['sigma_best_match'],
        linestyle='none', color='k', marker='o', label="BayesWave")


ax_mchirp.plot(cwb_result['best_match'],
        cwb_result['injected_chirp_mass']-cwb_result['best_chirp_mass'],
        linestyle='none', color='k', marker='s', label="CWB")

ax_mchirp.minorticks_on()
ax_mchirp.set_xlabel('Fitting Factor')
ax_mchirp.set_ylabel('Injected - Recovered Chirp Mass [M$_{\odot}$]')
ax_mchirp.legend()#loc='center left')
f_mchirp.tight_layout()

f_mchirp.savefig('chirpmass_fittingfactor.eps')





