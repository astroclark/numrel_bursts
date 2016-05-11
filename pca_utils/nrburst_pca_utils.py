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
bhex_utils.py
"""

import sys, os
import os.path
import subprocess
import cPickle as pickle

import numpy as np
import scipy.optimize
from scipy.spatial.distance import euclidean as euclidean_distance
from sklearn.decomposition import PCA 

import lal
import pycbc.types
import pycbc.filter
from pycbc.waveform import utils as wfutils

import nrburst_utils as nrbu

class catalog:
    """
    Contains attributes of catalog and numpy arrays with feature aligned
    waveforms
    """
    def __init__(self, simulations):

        print "Building catalogue"
        self.simulations = simulations
        self.amplitude_matrix, self.phase_matrix = build_catalog(simulations)

class bbh_pca:
    """
    Contains the PCA decomposition of an aligned catalog
    """

    def __init__(self, catalog, delta_t=1./1024, noise_file=None):

        self.delta_t = delta_t
        #
        # --- Peform the PCA decomposition
        #
        print "Performing PCA"
        self.pca = perform_pca(catalog.amplitude_matrix,
                catalog.phase_matrix)

        nsims = catalog.simulations.nsimulations

        #
        # --- Compute nominal projection coefficients and matches
        #
        self.amplitude_betas = np.zeros(shape=(nsims,nsims))
        self.phase_betas = np.zeros(shape=(nsims,nsims))

        self.amplitude_euclidean_distance = np.zeros(shape=(nsims,nsims))
        self.phase_euclidean_distance = np.zeros(shape=(nsims,nsims))
        self.matches = np.zeros(shape=(nsims,nsims), dtype=complex)

        if noise_file is not None:

            # Load an asd from file

            tmp = pycbc.types.TimeSeries(catalog.amplitude_matrix[0,:],
                    delta_t=delta_t)
            delta_f = tmp.to_frequencyseries().delta_f
            sample_frequencies = tmp.to_frequencyseries().sample_frequencies

            noise_data = np.loadtxt(noise_file)
            noise_asd = np.exp(np.interp(sample_frequencies, noise_data[:,0],
                np.log(noise_data[:,1])))

            self.noise_psd = pycbc.types.FrequencySeries(noise_asd**2, delta_f=delta_f)

        for w in xrange(nsims):

            projection = self.project_waveform(catalog.amplitude_matrix[w,:],
                catalog.phase_matrix[w,:])

            self.amplitude_betas[w,:] = np.copy(projection['amplitude_betas'])
            self.phase_betas[w,:] = np.copy(projection['phase_betas'])

            for n in xrange(nsims):


                # Reconstruct for each number of PCs
                recamp, recphase = self.reconstruct_ampphase(
                        catalog.amplitude_matrix[w,:],
                        catalog.phase_matrix[w,:], npcs=n+1)

                self.amplitude_euclidean_distance[w,n] = euclidean_distance(recamp,
                        catalog.amplitude_matrix[w,:])
                self.phase_euclidean_distance[w,n] = euclidean_distance(recphase,
                        catalog.phase_matrix[w,:])


                # Compute match with hplus
                hplus = pycbc.types.TimeSeries(np.real(catalog.amplitude_matrix[w,:] *
                        np.exp(1j*catalog.phase_matrix[w,:])), delta_t=self.delta_t)
                hplus_rec = \
                        pycbc.types.TimeSeries(np.real(recamp*np.exp(1j*recphase)),
                                delta_t=self.delta_t)

                plus_match , _ = pycbc.filter.match(hplus, hplus_rec,
                        low_frequency_cutoff=30.0, psd=self.noise_psd)

                # Compute match with hcross
                hcross = pycbc.types.TimeSeries(np.imag(catalog.amplitude_matrix[w,:] *
                        np.exp(1j*catalog.phase_matrix[w,:])), delta_t=self.delta_t)
                hcross_rec = \
                        pycbc.types.TimeSeries(np.imag(recamp*np.exp(1j*recphase)),
                                delta_t=self.delta_t)
                cross_match , _ = pycbc.filter.match(hcross, hcross_rec,
                        low_frequency_cutoff=30.0, psd=self.noise_psd)


                self.matches[w,n] = plus_match + 1j*cross_match



    def project_waveform(self, amplitude, phase):
        """
        Project the waveform amplitude and phase
        """

        projection = dict()
        projection['amplitude_betas'] = \
                np.concatenate(self.pca['amplitude_pca'].transform(amplitude))
        projection['phase_betas'] =  \
                np.concatenate(self.pca['phase_pca'].transform(phase))

        return projection

    def reconstruct_ampphase(self, amplitude, phase, npcs=1):
        """
        Reconstruct the amplitude and phase data in amplitude using npcs
        """
        
        # Get projection
        projection = self.project_waveform(amplitude, phase)

        rec_amplitude = np.zeros(shape=np.shape(amplitude))
        rec_phase = np.zeros(shape=np.shape(phase))

        # Sum contributions from PCs
        for i in xrange(npcs):

            rec_amplitude += \
                    projection['amplitude_betas'][i]*\
                    self.pca['amplitude_pca'].components_[i,:]

            rec_phase += \
                    projection['phase_betas'][i]*\
                    self.pca['phase_pca'].components_[i,:]

        # De-center the reconstruction
        rec_amplitude += self.pca['amplitude_pca'].mean_
        rec_phase += self.pca['phase_pca'].mean_

        return rec_amplitude, rec_phase

    def file_dump(self, pcs_filename=None):
        """
        Dump to binary for LAL
        """

        if pcs_filename is None:
            print >> sys.stderr, "ERROR: you must provide a name for catalogue \
file dumps"
            sys.exit(-1)

        #
        # Ascii Dump
        #
        for pca_attr in self.pca.keys():
            
            pcaObj = self.pca[pca_attr]

            #
            # Ascii
            #
            this_name_asc  = os.path.join(pcs_filename + "_" + pca_attr +
                    ".asc")
            print "Dumping to %s"%this_name_asc

            # First row contains the mean waveform
            dims = np.shape(pcaObj.components_)
            output_array  = np.zeros(shape=(dims[0]+1,dims[1]))

            output_array[0,:]  = pcaObj.mean_
            output_array[1:,:] = pcaObj.components_

            np.savetxt(this_name_asc, output_array)

            #
            # Binary
            #
            this_name_bin  = os.path.join(pcs_filename + "_" + pca_attr +
                    ".dat")
            print "Dumping to %s"%this_name_bin

            fp = open(this_name_bin, "wb")
            output_array.tofile(fp)
            fp.close()

        return 0



def perform_pca(amplitudes, phases):
    """
    Do PCA with ampnitude and phase parts of the complex waveforms in complex_catalogue
    """

    amp_pca = PCA()
    amp_pca.fit(amplitudes)

    phase_pca = PCA()
    phase_pca.fit(phases)

    pca={}

    pca['amplitude_pca'] = amp_pca
    pca['phase_pca'] = phase_pca


    return pca



def build_catalog(simulations, mtotal=100.0, nTsamples=1024, delta_t=1./1024):

    """
    Build the data matrix.
    """


    # Preallocation
    amp_cat = np.zeros(shape=(simulations.nsimulations, nTsamples))
    phase_cat = np.zeros(shape=(simulations.nsimulations, nTsamples))

    for s, sim in enumerate(simulations.simulations):
        
        print "Adding {0} to catalog ({1} of {2})".format(sim['wavefile'], s,
                simulations.nsimulations)

        # Extract waveform
        hp, hc = nrbu.get_wf_pols(sim['wavefile'], mtotal=mtotal,
                inclination=0.0, delta_t=delta_t, f_lower=30, distance=100)

        amp = wfutils.amplitude_from_polarizations(hp, hc)
        phase = wfutils.phase_from_polarizations(hp, hc) 

        # Normalise to unit norm
        amp /= np.linalg.norm(amp)


        # --- Populate time series catalogue (align peak amp to center)
        peakidx = np.argmax(amp.data)

        # Apply some smoothing to the start / end to get rid of remaining
        # small-number junk

        ampthresh=1e-2
        
        # After ringdown:
        amptmp = np.copy(amp.data)
        amptmp[:peakidx] = 1e10
        postmerger = np.argwhere(amptmp < ampthresh*max(amp.data))[0]
        
        win = lal.CreateTukeyREAL8Window(int(len(phase)-postmerger), 0.1)
        window = 1-win.data.data
        window[0.5*len(window):]=0.0
        phase.data[postmerger:] *= window
        amp.data[postmerger:] *= window
        
        # before waveform:
        premerger = np.argwhere(amp>ampthresh*max(amp.data))[0]
        
        win = lal.CreateTukeyREAL8Window(int(len(phase)-premerger), 0.1)

        window = win.data.data
        window[0.5*len(window):]=1.0

        phase.data *= window
        amp.data *= window
        
        # POPULATE
        # right
        start = 0.5*nTsamples
        end = start+len(amp.data)-peakidx
        amp_cat[s, start:end] = np.copy(amp.data[peakidx:])
        phase_cat[s, start:end] = np.copy(phase.data[peakidx:])

        # left
        start = 0.5*nTsamples-peakidx
        end = 0.5*nTsamples
        amp_cat[s, start:end] = np.copy(amp.data[:peakidx])
        phase_cat[s, start:end] = np.copy(phase.data[:peakidx])


        return (amp_cat, phase_cat)

    def main():
        print >> sys.stdout, sys.argv[0]
        print >> sys.stdout, __version__
        print >> sys.stdout, __date__
        return 0


