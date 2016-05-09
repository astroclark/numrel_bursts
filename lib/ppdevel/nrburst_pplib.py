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
nrburst_pplib.py

Post-process LIB-PCA posterior samples:
    * Corner plot
    * Matches with injection

"""

from __future__ import division
import sys, os
from optparse import OptionParser

import numpy as np
from matplotlib import pyplot as pl
import triangle

import lal
import lalmetaio
from glue.ligolw import lsctables, table, utils, ligolw, ilwd
table.use_in(ligolw.LIGOLWContentHandler)
lsctables.use_in(ligolw.LIGOLWContentHandler)

from pylal import bayespputils as bppu

import nrburst_utils as nrbu

def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-i", "--sim-inspiral", type=str, default=None)
    parser.add_option("-n", "--npcs", type=int, default=1)
    parser.add_option("-e", "--event", type=int, default=None)
    parser.add_option("-d", "--delta-t", type=float, default=1./512)
    parser.add_option("-L", "--datalen", type=float, default=4.0)

    (opts,args) = parser.parse_args()


    if len(args)==0:
        print >> sys.stderr, "Must supply a posterior samples file as a commandline argument"
        sys.exit()

    if not os.path.isfile(args[0]):
        print >> sys.stderr, "posterior samples file requested: %s does not exist"%args[0]
        sys.exit()


    if opts.sim_inspiral is not None:
        if not os.path.isfile(opts.sim_inspiral):
            print >> sys.stderr, "sim-inspiral file not found at: %s"%opts.sim_inspiral
            sys.exit()

        if opts.event is None:
            print >> sys.stderr, "must select event from sim-inspiral table (--event)"
            sys.exit()

    return opts, args


def copy_sim_inspiral( row ):
    """
    Turn a lsctables.SimInspiral into a SWIG wrapped lalburst.SimInspiral
    """
    swigrow = lalmetaio.SimInspiralTable()
    for simattr in lsctables.SimInspiralTable.validcolumns.keys():
        if simattr in ["waveform", "source", "numrel_data", "taper"]:
            # unicode -> char* doesn't work
            setattr( swigrow, simattr, str(getattr(row, simattr)) )
        else:
            setattr( swigrow, simattr, getattr(row, simattr) )
    # FIXME: This doesn't get copied properly, and so is done manually here.
    swigrow.geocent_end_time = lal.LIGOTimeGPS(row.geocent_end_time, row.geocent_end_time_ns)
    return swigrow

def get_sims( sim_inspiral_file ):
    """
    Return a list of swig-wrapped siminspiral table entries
    """

    xmldoc = utils.load_filename(sim_inspiral_file,
            contenthandler=ligolw.LIGOLWContentHandler)

    sims = []
    try:
        sim_insp = table.get_table( xmldoc,
                lsctables.SimInspiralTable.tableName )
        sims.extend( map(copy_sim_inspiral, sim_insp) )
    except ValueError:
        if opts.verbose:
            print >> sys.stderr, "No SimInspiral table found, \
skipping..."


    return sims


def main(opts,args):


    if opts.sim_inspiral is not None:
        # Build the injected waveform

        #
        # Parse sim_inspiral table
        #
        sims = get_sims(opts.sim_inspiral)
 
        # Sim_inspiral row for this event:
        sim = sims[opts.event]

        hp, hc = nrbu.get_wf_pols(sim.numrel_data, mtotal=sim.mass1+sim.mass2,
                inclination=sim.inclination, delta_t=1./1024, f_lower=30,
                distance=sim.distance)

        h1_signal = nrbu.project_waveform(hp, hc, skyloc=(sim.latitude,
            sim.longitude),polarization=sim.polarization, detector_name="H1")
#
#
#       sys.exit()
#
#
#       # Injection set object with details (and methods) for injections
#       injSet = pycbc.inject.InjectionSet(opts.sim_inspiral)
#
#       h1_epoch = sim.h_end_time + 1e-9*sim.h_end_time_ns \
#               -0.5*opts.datalen
#       h1_injection = pycbc.types.TimeSeries(
#               np.zeros(opts.datalen/opts.delta_t),
#               delta_t=opts.delta_t, epoch=h1_epoch
#               )
#       h1_injection = injSet.apply(h1_injection, 'H1')



    #
    # Load and parse posterior samples file
    #
    peparser = bppu.PEOutputParser('common')
    resultsObj = peparser.parse(open(args[0], 'r'))
    posterior = bppu.Posterior(resultsObj)

    return posterior, sim


if __name__ == "__main__":

    opts, args = parser()

    posterior, sim = main(opts,args)



