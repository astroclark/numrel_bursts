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
nrburst_doublescat.py

Summary plots for reconstruction analysis

"""

import os
import numpy as np
import nrburst_utils as nrbu
from matplotlib import pyplot as pl
pl.rcParams.update({'axes.labelsize': 16})
pl.rcParams.update({'xtick.labelsize':16})
pl.rcParams.update({'ytick.labelsize':16})
pl.rcParams.update({'legend.fontsize':16})

bw_path='GW150914_production/BayesWave/GW150914/GW150914_BW_FF_s1dotLs2dotL-q.npz'
bw_results=np.load(os.path.join(os.environ.get('HOME'), bw_path))

cwb_path='GW150914_production/CWB/GW150914/GW150914_CWB_FF_s1dotLs2dotL-q.npz'
cwb_results=np.load(os.path.join(os.environ.get('HOME'), cwb_path))

f, ax = nrbu.double_double_scatter_plot(
        param1y=(cwb_results['a1dotL'], bw_results['a1dotL']),
        param2y=(cwb_results['a2dotL'], bw_results['a2dotL']),
        paramx=(cwb_results['mass_ratios'], bw_results['mass_ratios']),
        matches=(cwb_results['median_matches'], bw_results['median_matches']),
        labelx='q',
        label1y=r'$\hat{\mathbf{S}}_1 . \hat{\mathbf{L}}$',
        label2y=r'$\hat{\mathbf{S}}_2 . \hat{\mathbf{L}}$')


f.savefig('GW150914_FF_s1dotLs2dotL-q.png')
f.savefig('GW150914_FF_s1dotLs2dotL-q.eps')

# Dump ascii data
fout = open('nrburst_reconstructionVsNR.txt','w')
fout.writelines("# a1dotL_CWB a1dotL_BW a2dotL_CWB a2dotL_BW mass_ratio_CWB mass_ratio_BW fittingfactor_CWB fittingfactor_BW\n")
for f in range(len(cwb_results['a1dotL'])):
    fout.writelines("{a1dotL_CWB} {a1dotL_BW} {a2dotL_CWB} {a2dotL_BW} {mass_ratio_CWB} {mass_ratio_BW} {fittingfactor_CWB} {fittingfactor_BW}\n".format(
        a1dotL_CWB=cwb_results['a1dotL'][f], a1dotL_BW=bw_results['a1dotL'][f], 
        a2dotL_CWB=cwb_results['a2dotL'][f], a2dotL_BW=bw_results['a2dotL'][f], 
        mass_ratio_CWB=cwb_results['mass_ratios'][f],
        mass_ratio_BW=bw_results['mass_ratios'][f],
        fittingfactor_CWB=cwb_results['median_matches'][f],
        fittingfactor_BW=bw_results['median_matches'][f]
        ))

fout.close()
 
