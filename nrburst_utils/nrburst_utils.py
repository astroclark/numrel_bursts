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
nrburst_utils.py

Module for loading and building catalogs from the GT_BBH_BURST_CATALOG

"""

from __future__ import division

import os
import sys 
import subprocess
from optparse import OptionParser
import ConfigParser
import glob
import operator

import h5py

import numpy as np
import scipy.signal as signal

import lal
import lalsimulation as lalsim
import pycbc.filter
import pycbc.types
from pycbc.waveform import get_td_waveform
from pycbc.waveform import utils as wfutils
from pycbc import pnutils
from pycbc.detector import Detector

__author__ = "James Clark <james.clark@ligo.org>"
#git_version_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
#__version__ = "git id %s" % git_version_id

gpsnow = subprocess.check_output(['lalapps_tconvert', 'now']).strip()
__date__ = subprocess.check_output(['lalapps_tconvert', gpsnow]).strip()

# *****************************************************************************
global __param_names__
#__param_names__ = ['D', 'mres', 'q', 'a1', 'a2', 'th1L', 'th2L', 'ph1', 'ph2', 'th12',
#                'thSL', 'thJL', 'Mmin30Hz', 'Mmin10Hz', 'Mchirpmin30Hz', 'a1x',
#                'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'Lx',  'Ly', 'Lz', 'mf', 'af']
#__param_names__ = ['D', 'q', 'a1', 'a2', 'th1L', 'th2L', 'ph1', 'ph2', 'th12',
#                'thSL', 'thJL', 'Mmin30Hz', 'Mmin10Hz', 'Mchirpmin30Hz', 'a1x',
#                'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'Lx',  'Ly', 'Lz', 'mf', 'af']
__param_names__ = ['Mchirpmin30Hz', 'Mmin30Hz', 'a1', 'a2', 'eta', 'q',
'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z']


# *****************************************************************************
# Contents 
#
# 1) General purpose signal processing tools
#       a) highpass()
#       b) window_wave()
#       c) planckwin()
#
# 2) Physics Functions
#       a) component_masses() 
#       b) cartesian_spins()
#       c) mtot_from_mchirp()
#
# 3) Match calculations
#       a) scale_wave()
#       b) mismatch()
#       c) parser()
#       d) configuration
#
# 4) Waveform catalog Tools
#       a) simulation_details
#       b) waveform_catalog

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# General purpose signal processing tools


def highpass(timeseries, delta_t=1./512, knee=9., order=12, attn=0.1):
    """
    Trivial interface function to make looping through catalogs neater
    """

    tmp = pycbc.types.TimeSeries(initial_array=timeseries, delta_t=delta_t)

    return np.array(pycbc.filter.highpass(tmp, frequency=knee, filter_order=order,
            attenuation=attn).data)

def window_wave(input_data):

    nonzero=np.argwhere(abs(input_data)>1e-3*max(abs(input_data)))
    idx = range(nonzero[0],nonzero[-1])
    win = planckwin(len(idx), 0.3)
    win[0.5*len(win):] = 1.0
    input_data[idx] *= win

    return input_data

def planckwin(N, epsilon):

    t1 = -0.5*N
    t2 = -0.5*N * (1.-2.*epsilon)
    t3 = 0.5*N * (1.-2.*epsilon)
    t4 = 0.5*N

    Zp = lambda t: (t2-t1)/(t-t1) + (t2-t1)/(t-t2)
    Zm = lambda t: (t3-t4)/(t-t3) + (t3-t4)/(t-t4)

    win = np.zeros(N)
    ts = np.arange(-0.5*N, 0.5*N)

    for n,t in enumerate(ts):
        if t<=t1:
            win[n] = 0.0
        elif t1<t<t2:
            win[n] = 1./(np.exp(Zp(t))+1)
        elif t2<=t<=t3:
            win[n] = 1.0
        elif t3<t<t4:
            win[n] = 1./(np.exp(Zm(t))+1)

    return win

def phase_of(z):
    return np.unwrap(np.angle(z))

def taper(input_data, delta_t):
    """ 
    Window out the inspiral (everything prior to the biggest peak)
    """

    timeseries = lal.CreateREAL8TimeSeries('blah', 0.0, 0,
            delta_t, lal.StrainUnit, int(len(input_data)))
    timeseries.data.data = np.copy(input_data)

    lalsim.SimInspiralREAL8WaveTaper(timeseries.data,
        lalsim.SIM_INSPIRAL_TAPER_START)
        #lalsim.SIM_INSPIRAL_TAPER_STARTEND)

    return timeseries.data.data

def extract_wave(inwave, datalen=4.0, sample_rate = 4096):
    """
    Extract a subsection of a reconstructed waveform
    """
    extract_len = 0.5 # retain this many seconds of reconstruction
    delta = 0.15 # center the retained data on the peak of the waveform minus
                 # this many seconds
    peakidx = np.argmax(abs(inwave)) - delta*sample_rate
    nsamp = extract_len * sample_rate

    extracted = inwave[int(peakidx-0.5*nsamp): int(peakidx+0.5*nsamp)]

    win = lal.CreateTukeyREAL8Window(len(extracted), 0.1)
    extracted *= win.data.data

    output = np.zeros(datalen*sample_rate)
    output[0.5*datalen*sample_rate-0.5*nsamp:
            0.5*datalen*sample_rate+0.5*nsamp] = np.copy(extracted)

    return output

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Physics Functions

def component_masses(total_mass, mass_ratio):
    """
    Return m1 and m2, given total mass and mass ratio (m1/m2)

    m1, m2 = component_masses(total_mass, mass_ratio)
    """

    m1 = mass_ratio * total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1

    return m1, m2


def cartesian_spins(spin_magnitude, spin_theta):
    """
    Compute cartesian spin components.  Only does z-component for now
    """

    if np.isnan(spin_magnitude) or np.isnan(spin_theta):
        return 0.0
    else:
        spin_z = spin_magnitude * np.cos(spin_theta * np.pi / 180.0)
    return spin_z

def mtot_from_mchirp(mc, q):
    """
    Compute the total mass from chirp mass and mass ratio (m1/m2)
    """
    eta = q/(1+q)**2.0
    return mc * eta**(-3./5)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match calculations

def scale_wave(wave, target_total_mass, init_total_mass):
    """
    Scale the waveform to total_mass.  Assumes the waveform is initially
    generated at init_total_mass defined in this script.
    """
    scaling_data = np.copy(wave.data[:])

    amp = abs(scaling_data)

    scale_ratio = target_total_mass / init_total_mass
    scaling_data *= scale_ratio

    peakidx = np.argmax(amp)

    interp_times = scale_ratio * wave.sample_times.data[:] - \
            peakidx*wave.delta_t*(scale_ratio-1)

    resampled_wave = np.interp(wave.sample_times.data[:], interp_times,
            scaling_data)

    return resampled_wave

def get_wf_pols(file, mtotal, inclination=0.0, delta_t=1./1024, f_lower=30,
        distance=100):
    """
    Generate the NR_hdf5_pycbc waveform from the HDF5 file <file> with specified
    params
    """

    f = h5py.File(file, 'r')


    # Metadata parameters:

    params = {}
    params['mtotal'] = mtotal

    params['eta'] = f.attrs['eta']

    params['mass1'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[0]
    params['mass2'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[1]

    params['spin1x'] = f.attrs['spin1x']
    params['spin1y'] = f.attrs['spin1y']
    params['spin1z'] = f.attrs['spin1z']
    params['spin2x'] = f.attrs['spin2x']
    params['spin2y'] = f.attrs['spin2y']
    params['spin2z'] = f.attrs['spin2z']

    params['coa_phase'] = f.attrs['coa_phase']

    f.close()

    hp, hc = get_td_waveform(approximant='NR_hdf5_pycbc', 
                                     numrel_data=file,
                                     mass1=params['mass1'],
                                     mass2=params['mass2'],
                                     spin1x=params['spin1x'],
                                     spin1y=params['spin1y'],
                                     spin1z=params['spin1z'],
                                     spin2x=params['spin2x'],
                                     spin2y=params['spin2y'],
                                     spin2z=params['spin2z'],
                                     delta_t=delta_t,
                                     f_lower=f_lower,
                                     inclination=inclination,
                                     coa_phase=params['coa_phase'],
                                     distance=distance)


    hp_tapered = wfutils.taper_timeseries(hp, 'TAPER_START')
    hc_tapered = wfutils.taper_timeseries(hc, 'TAPER_START')

    return hp_tapered, hc_tapered

def project_waveform(hp, hc, skyloc=(0.0, 0.0), polarization=0.0, detector_name="H1"):
    """
    Project the hp,c polarisations onto detector detname for sky location skyloc
    and polarisation pol
    """

    detector = Detector(detector_name)

    signal = detector.project_wave(hp, hc, skyloc[0], skyloc[1],
            polarization)

    return signal


def mismatch(params,
        skyloc=(0,0), nrfile=None, detector_name="H1", mass_bounds=None,
        rec_data=None, asd=None, delta_t=1./1024, f_min=30.0):
    """
    Compute mismatch (1-match) between the tmplt wave and the event wave, given
    the total mass.  Uses rec_data and psd which are defined globally in the
    calling script.

    Note: the reconstructed waveform which is passed in should be the whitened
    detector response, so that the template waveform is whitened by the ASD
    prior to the match calculation, and no PSD is passed directly to match().

    XXX: Can't i just pass in the config object to get the fixed params
    """
    mtotal, inclination, polarization = params
#   mtotal = float(params)
#   inclination = 0.0
#   polarization = 0.0
#
    min_mass, max_mass = mass_bounds

    if (mtotal >= min_mass) and (mtotal <= max_mass):

        # Generate the polarisations
        hp, hc = get_wf_pols(nrfile, mtotal, inclination=inclination, delta_t=delta_t)

        # Project to detector
        tmplt = project_waveform(hp, hc, skyloc=skyloc,
                polarization=polarization, detector_name=detector_name)

        # Put the reconstruction data in a TimeSeries
        rec_data = pycbc.types.TimeSeries(rec_data, delta_t=delta_t)

        # Resize to the same length as the data
        tlen = max(len(tmplt), len(rec_data))
        tmplt.resize(tlen)
        rec_data.resize(tlen)


        # Whiten the template
        Tmplt = tmplt.to_frequencyseries()
        Tmplt.data /= asd

        try:
            match, _ = pycbc.filter.match(Tmplt, rec_data, psd=None,
                    low_frequency_cutoff=f_min)
        except ZeroDivisionError:
            match = np.nan

        return 1-match

    else:
        # Outside of mass range

        return 1.


def parser():
    """
    Parser for match calculations
    """

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-t", "--user-tag", default="TEST", type=str)
    parser.add_option("-o", "--output-dir", type=str, default=None)
    parser.add_option("-a", "--algorithm", type=str, default=None)

    (opts,args) = parser.parse_args()

    if len(args)==0:
        print >> sys.stderr, "ERROR: require config file"
        sys.exit()

    algorithms=["BW", "CWB"]
    if opts.algorithm is not None and opts.algorithm not in algorithms:
        print >> sys.stderr, "ERROR: algorithm %s not recognised"%opts.algorithm
        print >> sys.stderr, "must be in ", algorithms
        sys.exit(-1)


    # --- Read config file
    configparser = ConfigParser.ConfigParser()
    configparser.read(args[0])

    # --- Where did the reconstruction come from?
    if opts.algorithm is not None:
        # override from the commandline
        configparser.set('analysis','algorithm',opts.algorithm)

    # Check algorithm is defined (might have been in the ini file)
    if configparser.has_option('analysis', 'algorithm'):
        alg = configparser.get('analysis','algorithm')
        if alg not in algorithms:
            print >> sys.stderr, "ERROR: algorithm %s not recognised"%alg
            print >> sys.stderr, "must be in ", algorithms
            sys.exit(-1)
    else:
        print >> sys.stderr, "ERROR: algorithm not defined"
        print >> sys.stderr, "must be in ", algorithms
        print >> sys.stderr, "and defined in [analysis] of ini or with --algorithm"
        sys.exit(-1)


    return opts, args, configparser

class configuration:
    """
    Class for configuration details.  Easy to keep a record of what was run
    """

    def __init__(self, configparser):

        self.sample_rate=configparser.getint('analysis', 'sample_rate')
        self.delta_t=1./self.sample_rate
        self.datalen=configparser.getfloat('analysis', 'datalen')
        self.f_min=configparser.getfloat('analysis', 'f_min')
        self.algorithm=configparser.get('analysis', 'algorithm')
        self.detector_name=configparser.get('analysis', 'detector_name')

        self.nsampls=configparser.getint('parameters', 'nsampls')
        self.mass_guess=configparser.getfloat('parameters', 'mass_guess')
        self.min_chirp_mass=configparser.getfloat('parameters', 'min_chirp_mass')
        self.max_chirp_mass=configparser.getfloat('parameters', 'max_chirp_mass')

        self.reconstruction=configparser.get('paths', 'reconstruction')
        self.spectral_estimate=configparser.get('paths', 'spectral-estimate')
        self.catalog=configparser.get('paths', 'catalog')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform catalog Tools

                     
class simulation_details:
    """
    The waveform catalog for the chosen series (possibly plural) with
    user-specified parameter bounds.
    
    Example Usage:

    In [24]: bounds = dict()

    In [25]: bounds['q'] = [2, np.inf]

    In [26]: simcat = bwave.simulation_details(param_bounds=bounds)

    In [28]: for sim in simcat.simulations: print sim
    {'a1': 0.6, 'th2L': 90.0, 'D': 7.0, 'thJL': 19.8, 'th1L': 90.0, 'q': 2.5, 'th12': 180.0, 'a2': 0.6,  'wavefile': ['/home/jclark308/Projects/bhextractor/data/NR_data/GT_BBH_BURST_CATALOG/Eq-series/Eq_D7_q2.50_a0.6_ph270_m140/Strain_jinit_l2_m2_r75_Eq_D7_q2.50_a0.6_ph270_m140.asc'], 'wavename': 'Eq_D7_q2.50_a0.6_ph270_m140', 'ph2': 90.0, 'ph1': -90.0, 'Mmin30Hz': 97.3, 'Mmin10Hz': 292.0, 'thSL': 90.0}

    ... and so on ...

    Note: will default to waveforms with min mass = 100 Msun permissable for a
    low-frequency-cutoff at 30 Hz, unless Mmin10Hz or a different minimum mass
    is defined

    """

    def __init__(self, param_bounds=None, catdir=None, fmin=30.0):

        # ######################################################
        # Other initialisation
        self.param_bounds = param_bounds

        self.fmin = fmin
        self.catdir=catdir

        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "Finding matching waveforms"

        # Get all waveforms
        #self.simulations = self.list_simulations(series_names)
        self.simulations = self.list_simulations(catdir=self.catdir)
        self.nsimulations = len(self.simulations)

        print "----"
        print "Found %d waveforms matching criteria:"%(self.nsimulations)
        print "Bounds: ", param_bounds
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    def list_simulations(self, catdir=None):
        """
        Creates a list of simulation dictionaries which contain the locations
        of the data files and the physical parameters the requested series
        """

        if catdir is None:
            print >> sys.stderr, "ERROR: must specify directory of NR waveforms"
            sys.exit(-1)

        readme_file = os.path.join(catdir, 'README.txt')

        # Get all simulations (from readme)
        simulations = self._get_series(catdir, readme_file)

        # Down-select on parameters
        if self.param_bounds is not None:
            for bound_param in self.param_bounds.keys():
                simulations = self._select_param_values(simulations,
                        bound_param, self.param_bounds[bound_param])

        simulations = self._check_sim_unique(simulations)

        return simulations

    @staticmethod
    def _check_sim_unique(simulations):
        """
        Reduce the list of simulations to those which have unique parameter
        combinations

        XXX: Note that this (currently) takes the FIRST unique simulation; it
        doesn't care what the resolution or series was
        """
        print "Ensuring uniqueness of simulations"


        # Make a copy of the list of simulations
        unique_simulations = list(simulations)

        #physical_params = 'q', 'a1', 'a2', 'th1L', 'th2L', 'ph1', 'ph2', \
        #        'th12', 'thSL', 'thJL', 'mres'
        #physical_params = 'q', 'a1', 'a2', 'th1L', 'th2L', 'ph1', 'ph2', \
        #        'th12', 'thSL', 'thJL'
        physical_params = 'a1', 'q', 'spin2x', 'spin2y', 'spin2z', 'eta', \
                'spin1y', 'spin1x', 'spin1z', 'a2'

        param_sets = []

        # Loop through each simulation
        for s in xrange(len(simulations)):

            # array of physical parameter values
            param_vals = np.zeros(len(physical_params))
            for p,param_name in enumerate(physical_params):
                param_vals[p] = simulations[s][param_name]
            param_vals[np.isnan(param_vals)] = np.inf

            # Create a tuple with the parameter values
            param_sets.append(tuple(param_vals))

        unique_param_sets = list(set(param_sets))
 
 
        # Now loop through the unique sets 
        for unique_param_set in unique_param_sets:
 
            # Identify indices for parameter sets of parameter values which
            # occur in the unique sets of parameter values
            indices = [i for i, x in enumerate(param_sets) if [x] ==
                    [unique_param_set]]
 
            if len(indices)>1:
 
                resorted_simulations = list(np.array(simulations)[indices])
                resorted_simulations.sort(key=operator.itemgetter('Mmin30Hz'))
 
                # Then there are multiple simulations with the same set of
                # parameters - we need to remove all but 1
 
                #for index in indices[1:]:
                print '----'
                print "retaining ", resorted_simulations[0]['wavename']
                for sim in resorted_simulations[1:]:
                    # Remove everything after the first simulation which has
                    # this parameter set
                    print "removing ", sim['wavename']
                    unique_simulations.remove(sim)

        return unique_simulations

    @staticmethod
    def _select_param_values(simulations, param, bounds):
        """
        Return the list of simulations with parameter values in [low_bound,
        upp_bound]
        """
        return [sim for sim in simulations if sim[param] >= min(bounds) and
                sim[param]<=max(bounds) ]

    @staticmethod
    def _get_series(datadir, readme_file):
        """
        Read the parameters from the readme file and return a list of simulation
        dictionaries
        """
        readme_data = np.loadtxt(readme_file, dtype=str)
        
        simulations = []
        #nNotFound = 0
        for s in xrange(len(readme_data)):

            sim = dict()

            runID = readme_data[s,0]
            sim['wavefile'] = readme_data[s,1]

            #wavename = readme_data[s,1]
            #wavefile = glob.glob(os.path.join(datadir, runID, '*asc'))
            #wavefile = glob.glob(os.path.join(datadir, runID, '*h5'))

            # Check that this waveform exists
#           if len(wavefile)>1:
#               print >> sys.stderr, "Error, more than one data file in directory: %s"%(
#                       os.path.join(datadir,runID))
#               sys.exit(-1)
#           elif len(wavefile)==0:
#               print >> sys.stderr, "WARNING, No file matching glob pattern: \n%s"%(
#                       os.path.join(datadir,runID, '*h5'))
#               nNotFound+=1
#                continue

            #sim['wavename'] = wavename
            #sim['wavefile'] = wavefile[0]
            sim['runID'] = int(runID)

            start = len(readme_data[s,:]) -  len(__param_names__)
            param_vals = [float(param) for param in readme_data[s,start:]]

            # physical params
            for p,param_name in enumerate(__param_names__):
                sim[param_name] = param_vals[p]

            simulations.append(sim)


        return simulations


# *******************************************************************************
def main():
    print >> sys.stdout, sys.argv[0]
    print >> sys.stdout, __version__
    print >> sys.stdout, __date__
    return 0
