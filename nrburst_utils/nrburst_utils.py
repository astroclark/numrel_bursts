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
bhextractor_wavedata.py

Module for loading and building catalogs from the GT_BBH_BURST_CATALOG

"""

from __future__ import division

import os
import sys 
import subprocess
from optparse import OptionParser
import ConfigParser
import glob
import cPickle as pickle
import operator

import numpy as np
import scipy.signal as signal

import lal
import lalsimulation as lalsim
import pycbc.filter
import pycbc.types

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
__param_names__ = ['D', 'q', 'a1', 'a2', 'th1L', 'th2L', 'ph1', 'ph2', 'th12',
                'thSL', 'thJL', 'Mmin30Hz', 'Mmin10Hz', 'Mchirpmin30Hz', 'a1x',
                'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'Lx',  'Ly', 'Lz', 'mf', 'af']

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


def mismatch(target_total_mass, init_total_mass, mass_bounds, tmplt_wave_data,
        event_wave_data, asd=None, delta_t=1./512, delta_f=0.25,
        ifo_response=False, f_min=30.0):
    """
    Compute mismatch (1-match) between the tmplt wave and the event wave, given
    the total mass.  Uses event_wave and psd which are defined globally in the
    calling script.

    XXX: Planned revision - pass in a params dictionary for the pycbc NR
    waveform infrastructure and generate the wavefrom from that.
    """
    min_mass, max_mass = mass_bounds

    if (target_total_mass >= min_mass) and (target_total_mass <= max_mass):

        # Convert the real part of the wave to a pycbc timeseries object
        init_tmplt = pycbc.types.TimeSeries(np.real(tmplt_wave_data[:]),
                delta_t=delta_t)

        # Rescale the template to this total mass
        tmplt = pycbc.types.TimeSeries(scale_wave(init_tmplt, target_total_mass,
            init_total_mass), delta_t=delta_t)

        if ifo_response and asd is not None:
            # Whiten the template
            Tmplt = tmplt.to_frequencyseries()
            Tmplt.data /= asd

            # IFFT (just simplifies the code below) 
            tmplt = Tmplt.to_timeseries()


        if asd is not None and not ifo_response:
            psd = pycbc.types.FrequencySeries(asd**2, delta_f=delta_f)
        else:
            psd = None

        # Put the reconstruction data in a TimeSeries
        event_wave = pycbc.types.TimeSeries(event_wave_data, delta_t=delta_t)

        try:
            match, _ = pycbc.filter.match(tmplt, event_wave, psd=psd,
                    low_frequency_cutoff=f_min)
        except ZeroDivisionError:
            match = np.nan

        return 1-match

    else:

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
        self.deltaT=1./self.sample_rate
        self.datalen=configparser.getfloat('analysis', 'datalen')
        self.f_min=configparser.getfloat('analysis', 'f_min')
        self.algorithm=configparser.get('analysis', 'algorithm')

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
        physical_params = 'q', 'a1', 'a2', 'th1L', 'th2L', 'ph1', 'ph2', \
                'th12', 'thSL', 'thJL'

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
        nNotFound = 0
        for s in xrange(len(readme_data)):

            sim = dict()

            runID = readme_data[s,0]
            wavename = readme_data[s,1]
            wavefile = glob.glob(os.path.join(datadir, runID, '*asc'))
            #wavefile = glob.glob(os.path.join(datadir, runID, '*h5'))

            # Check that this waveform exists
            if len(wavefile)>1:
                print >> sys.stderr, "Error, more than one data file in directory: %s"%(
                        os.path.join(datadir,runID))
                sys.exit(-1)
            elif len(wavefile)==0:
                print >> sys.stderr, "WARNING, No file matching glob pattern: \n%s"%(
                        os.path.join(datadir,runID, '*asc'))
                nNotFound+=1
                continue

            sim['wavename'] = wavename
            sim['wavefile'] = wavefile[0]
            sim['runID'] = int(runID)

            start = len(readme_data[s,:]) -  len(__param_names__)
            param_vals = [float(param) for param in readme_data[s,start:]]

            # physical params
            for p,param_name in enumerate(__param_names__):
                sim[param_name] = param_vals[p]

            simulations.append(sim)

        return simulations

class waveform_catalog:
    """
    This contains the waveform data (as +,x, Amp and phase time-series and
    F-domain waveforms) constructed from the catalog contained in a
    simulation_details class

    """

    def __init__(self, simulation_details, ref_mass=None, distance=1,
            SI_deltaT=1./1024, SI_datalen=4, NR_deltaT=0.1, NR_datalen=10000,
            trunc_time=False): 
        """
        Initalise with a simulation_details object and a reference mass ref_mass to
        which waveforms get scaled.  If ref_mass is None, waveforms are left in
        code units

        """

        self.simulation_details = simulation_details
        self.NR_deltaT = NR_deltaT
        self.NR_datalen = NR_datalen

        self.trunc_time = trunc_time
        # Load in the NR waveform data
        self.load_wavedata()


        # Produce physical catalog if reference mass specified
        if ref_mass is not None:

            # catalog_to_SI() adds variables to self

            self.catalog_to_SI(ref_mass=ref_mass, SI_deltaT=SI_deltaT,
                    distance=distance, SI_datalen=SI_datalen)


            # assign some characteristics which will be useful elsewhere (e.g.,
            # making PSDs)
            self.SI_deltaT = SI_deltaT

            example_ts = \
                    pycbc.types.TimeSeries(np.real(self.SIComplexTimeSeries[0,:]),
                            delta_t=self.SI_deltaT)
            example_fs = example_ts.to_frequencyseries()
            self.SI_deltaF = example_fs.delta_f
            self.SI_flen = len(example_fs)
            del example_ts, example_fs


    def load_wavedata(self):
        """
        Load the waveform data pointed to by the simulation_details object
        """

        # Load the waveform data into lists for time, plus and cross.  When all
        # data is loaded, identify the longest waveform and insert all waveforms
        # into a numpy array of that length

        time_data  = []
        plus_data  = []
        cross_data = []

        # Build waveforms in this catalog
        max_time=-np.inf

        for w, sim in enumerate(self.simulation_details.simulations):

            print 'Loading %s waveform (runID %d)'%(sim['wavename'], sim['runID'])

            wavedata = np.loadtxt(sim['wavefile'])

            time_data.append(wavedata[:,0])
            plus_data.append(wavedata[:,1])
            cross_data.append(wavedata[:,2])

        # Now resample to a consistent deltaT
        time_data_resampled  = []
        plus_data_resampled  = []
        cross_data_resampled = []

        print 'Resampling to uniform rate'
        for w in xrange(self.simulation_details.nsimulations): 
            deltaT = np.diff(time_data[w])[0]
            if deltaT != self.NR_deltaT:
                resamp_len = deltaT / self.NR_deltaT * len(plus_data[w])
                plus_data_resampled.append(signal.resample(plus_data[w],
                    resamp_len))
                cross_data_resampled.append(signal.resample(cross_data[w],
                    resamp_len))
            else:
                plus_data_resampled.append(np.copy(plus_data[w]))
                cross_data_resampled.append(np.copy(cross_data[w]))

        #
        # Insert into a numpy array
        #
        NR_nsamples = self.NR_datalen / self.NR_deltaT
        self.NRComplexTimeSeries = np.zeros(shape=(len(plus_data_resampled),
            NR_nsamples), dtype=complex)

        # Alignment & Normalisation
        align_idx = 0.5*NR_nsamples

        trunc_epsilon = 1e-3
        trunc_len = np.inf
        for w in xrange(self.simulation_details.nsimulations):

            # XXX
            wave = plus_data_resampled[w] - 1j*cross_data_resampled[w]

            # Normalisation (do this in the PCA)
            #wave /= np.vdot(wave,wave)

            peak_idx=np.argmax(abs(wave))
            start_idx = align_idx - peak_idx

            self.NRComplexTimeSeries[w,start_idx:start_idx+len(wave)] = wave

        del time_data, plus_data, cross_data, plus_data_resampled, cross_data_resampled

        return 0

    def catalog_to_SI(self, ref_mass, SI_deltaT=1./512, distance=1.,
            SI_datalen=4):
        """
        Convert waveforms in self.NRdata to physical time stamps / amplitudes
        """

        print "converting NR catalog to SI"

        # Add physical attributes
        self.ref_mass = ref_mass
        self.SI_deltaT=SI_deltaT
        self.distance=distance

        # Time steps of NR waveforms at the reference mass: 
        SI_deltaT_of_NR = ref_mass * lal.MTSUN_SI * self.NR_deltaT
        Mscale = ref_mass * lal.MRSUN_SI / (distance * 1e6 * lal.PC_SI)

        # Resample to duration in seconds (=nsamples x deltaT) x samples / second
        resamp_len = int(np.ceil(self.NR_datalen/self.NR_deltaT\
                *SI_deltaT_of_NR/self.SI_deltaT))

        self.SIComplexTimeSeries = \
                np.zeros(shape=(self.simulation_details.nsimulations,
                    SI_datalen/SI_deltaT), dtype=complex)

        # length of dummy series to position peaks halfway along - we will then
        # apply a Tukey window of SI_datalen to that time series
        SI_biglen = 32.0
        tukeywin = lal.CreateTukeyREAL8Window(resamp_len, 0.25)
        for w in xrange(self.simulation_details.nsimulations):

            # Resample to the appropriate length
            resampled_re = signal.resample(np.real(self.NRComplexTimeSeries[w,:]), resamp_len)
            resampled_im = signal.resample(np.imag(self.NRComplexTimeSeries[w,:]), resamp_len)

            # The complex wave
            wave = resampled_re - 1j*resampled_im

            # Apply a Tukey win to ensure smoothness
            wave*=tukeywin.data.data
            
            # Populate the center SI_datalen seconds of a zero-array of
            # SI_biglen seconds
            bigwave = np.zeros(SI_biglen / SI_deltaT, dtype=complex)

            # Locate peak of waveform and place it at the center of the big
            # array of zeros
            peakidx = np.argmax(abs(wave))
            startidx = 0.5*SI_biglen/SI_deltaT - peakidx

            bigwave[startidx:startidx+len(wave)] = \
                    Mscale*wave

            startidx = 0.5*SI_biglen/SI_deltaT - 0.5*SI_datalen/SI_deltaT
            self.SIComplexTimeSeries[w,:] = \
                    bigwave[startidx:startidx+SI_datalen/SI_deltaT]

        return 0

# *******************************************************************************
def main():
    print >> sys.stdout, sys.argv[0]
    print >> sys.stdout, __version__
    print >> sys.stdout, __date__
    return 0
