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
#global __param_names__
#__param_names__ = ['Mchirpmin30Hz', 'Mmin30Hz', 'a1', 'a2', 'eta', 'q',
#'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z']

global __metadata_ndecimals__ # number of decimal places to retain in metadata
__metadata_ndecimals__ = 3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# General purpose signal processing tools


def extract_wave(inwave, datalen=4.0, sample_rate = 4096):
    """
    Extract a subsection of a reconstructed waveform; useful for cleaning CWB
    reconstruction
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
# Derived physical quantities

def a_vec(spinx, spiny, spinz):

    a = np.array([spinx, spiny, spinz])
    a = np.around(a,decimals=__metadata_ndecimals__)
    anorm = np.linalg.norm(a)

    return a, anorm

def a_with_L(spinx, spiny, spinz):
    """
    Return the alignment of spin vector with angular momentum:
    
    dot([spinx, spiny, spinz], [0, 0, 1]), cross(...)
    """
    a, anorm = a_vec(spinx, spiny, spinz)
    Lhat = np.array([0,0,1])

    return np.dot(a, Lhat), np.cross(a, Lhat)


def effspin_with_L(mass_ratio, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z):
    """
    Return dot and cross products of effective spin vector with angular momentum
    """

    L_hat = np.array([0, 0, 1])

    mass1 = mass_ratio / (1.0+mass_ratio)
    mass2 = 1-mass1
    
    a1, a1norm = a_vec(spin1x, spin1y, spin1z)
    if a1norm==0:
        return 0.0, 0.0

    a2, a2norm = a_vec(spin2x, spin2y, spin2z)
    if a2norm==0:
        return 0.0, 0.0

    S1 = a1*mass1**2
    S2 = a2*mass2**2
    S_eff = (1.0 + 1.0/mass_ratio)*S1 + (1.0+mass_ratio)*S2
    S_effnorm = np.linalg.norm(S_eff)

    if S_effnorm > 0:

        return np.dot(S_eff, L_hat), np.cross(S_eff, L_hat)
    else:
        return 0.0, 0.0

def spin_angle(spin1x, spin1y, spin1z, spin2x, spin2y, spin2z):
    """
    Return angle (in degrees) subtended by spin vectors
    """
    a1, a1norm = a_vec(spin1x, spin1y, spin1z)
    if a1norm==0:
        return 0.0
    a1 /= a1norm

    a2, a2norm = a_vec(spin2x, spin2y, spin2z)
    if a2norm==0:
        return 0.0
    a2 /= a2norm

    if np.around(np.dot(a1,a2), decimals=__metadata_ndecimals__)==1:
        theta12 = 0.0
    elif np.around(np.dot(a1,a2), decimals=__metadata_ndecimals__)==-1:
        theta12 = lal.PI
    else:
        theta12 = np.arccos(np.dot(a1, a2))

    return theta12 / lal.PI_180


def totspin_dot_L(mass_ratio, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z):
    """
    Return dot product and angle between total spin and angular momentum
    """

    mass1 = mass_ratio / (1.0+mass_ratio)
    mass2 = 1-mass1

    a1 = np.array([spin1x, spin1y, spin1z])
    a2 = np.array([spin2x, spin2y, spin2z])

    # XXX: hdf5 metadata is apparently imprecise
    a1 = np.around(a1,decimals=__metadata_ndecimals__)
    a2 = np.around(a2,decimals=__metadata_ndecimals__)

    S1 = a1*mass1**2
    S2 = a2*mass2**2
    S = S1 + S2

    if np.linalg.norm(S) > 0.0:

        L_hat = np.array([0, 0, 1])
        SdotL = np.dot(S, L_hat)/np.linalg.norm(S)

        return SdotL, np.arccos(SdotL) / lal.PI_180
    else:
        return 0.0, 0.0



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match calculations


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
        skyloc=(0,0), polarization=0, nrfile=None, detector_name="H1",
        mass_bounds=None, rec_data=None, asd=None, delta_t=1./1024, f_min=30.0):
    """
    Compute mismatch (1-match) between the tmplt wave and the event wave, given
    the total mass.  Uses rec_data and psd which are defined globally in the
    calling script.

    Note: the reconstructed waveform which is passed in should be the whitened
    detector response, so that the template waveform is whitened by the ASD
    prior to the match calculation, and no PSD is passed directly to match().

    XXX: Can't i just pass in the config object to get the fixed params
    """
    mtotal, inclination = params

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
    parser.add_option("-s", "--simulation-number", type=str, default="all")

    (opts,args) = parser.parse_args()

    if opts.simulation_number != "all":
        print >> sys.stdout, "Analysis restricted to simulation %s"%(
                opts.simulation_number)
        opts.simulation_number = int(opts.simulation_number)

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

        self.sample_rate=configparser.getint('analysis', 'sample-rate')
        self.delta_t=1./self.sample_rate
        self.datalen=configparser.getfloat('analysis', 'datalen')
        self.f_min=configparser.getfloat('analysis', 'f-min')
        self.algorithm=configparser.get('analysis', 'algorithm')
        self.detector_name=configparser.get('analysis', 'detector-name')

        self.nsampls=configparser.getint('parameters', 'nsampls')
        self.min_chirp_mass=configparser.getfloat('parameters', 'min-chirp-mass')
        self.max_chirp_mass=configparser.getfloat('parameters', 'max-chirp-mass')

        self.reconstruction=configparser.get('paths', 'reconstruction')
        self.spectral_estimate=configparser.get('paths', 'spectral-estimate')
        self.catalog=configparser.get('paths', 'catalog')
        self.extrinsic_params=configparser.get('paths', 'extrinsic-params')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform catalog Tools

                     
class simulation_details:
    """
    The waveform catalog for the chosen series (possibly plural) with
    user-specified parameter bounds.
    
    Example bounds:

    In [24]: bounds = dict()

    In [25]: bounds['q'] = [2, np.inf]

    """

    def __init__(self, param_bounds=None, catdir=None):

        # ######################################################
        # Other initialisation
        self.param_bounds = param_bounds

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

        # --- Extract parameter names from readme file
        global __param_names__ 
        f = open(readme_file, 'r')
        __param_names__ = f.readline().split()[1:] # get rid of '#' 
        # Now get rid of runID and wavefile
        __param_names__.remove('runID')
        __param_names__.remove('wavefile')


        # Get all simulations (from readme)
        simulations = self._get_metadata(catdir, readme_file)

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

        physical_params = list(__param_names__)
        physical_params.remove('Mmin30Hz')
        physical_params.remove('Mchirpmin30Hz')

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
                print "retaining ", resorted_simulations[0]['wavefile']
                print resorted_simulations[0]
                for sim in resorted_simulations[1:]:
                    # Remove everything after the first simulation which has
                    # this parameter set
                    print "removing ", sim['wavefile']
                    print sim
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
    def _get_metadata(datadir, readme_file):
        """
        Read the parameters from the readme file and return a list of simulation
        dictionaries
        """
        readme_data = np.loadtxt(readme_file, dtype=str)

        # Get param names
        
        simulations = []
        for s in xrange(len(readme_data)):

            sim = dict()

            runID = readme_data[s,0]
            sim['wavefile'] = readme_data[s,1]
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
